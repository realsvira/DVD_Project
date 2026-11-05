"""Improvement phase, final step: ONE-TIME before/after evaluation at every level.

BEFORE = trained_models/final_before_improvement/  (models used in Parts A2b, B4-5, C)
AFTER  = trained_models/v3/                         (selected on validation only, d2_experiments.py)

Evaluation data
  PRIMARY (never used for any decision; generated for this phase, seen only here):
    data/final_eval/transistor_uniform   1500/case, spec distribution (salt 4)
    data/final_eval/transistor_targeted  1000/case, near-rail / small-|Vds| (salt 5)
    data/final_eval/gates                200 NEW corners (seed 5005) x 18 cases, gate SPICE refs
    data/final_eval/circuit              CLA3 at the new corners, 64 vectors x 200
  SECONDARY (already seen in earlier parts; reported for continuity, labelled as such):
    data/transistor_test2_uniform / _targeted, data/gates (3,600), data/circuit (12,800)

Outputs: results/improve/final/before_after.csv/.md, per-level tables, parity plots,
         final_eval_record.json (guard: refuses a second run without --force)

Run:  .\\.venv\\Scripts\\python.exe scripts\\d3_final_eval.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from leakage.circuit import build_cla3, predict_circuit_ml  # noqa: E402
from leakage.config import load_config  # noqa: E402
from leakage.gate_ml import GateLeakageModel, predict_gate_records  # noqa: E402
from leakage.inference import LeakagePredictor  # noqa: E402
from leakage.ml import INPUTS, TARGETS, evaluate  # noqa: E402

CFG = load_config()
OUT = CFG.results / "improve" / "final"
RECORD = OUT / "final_eval_record.json"
REFS = "original"
MODEL_SETS = {"before": CFG.trained_models / "final_before_improvement", "after": CFG.trained_models / "v3"}
KEY = ["gate", "inputs", "corner_id"]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_tr(d: Path, case: str) -> pd.DataFrame:
    df = pd.read_csv(d / f"{case}.csv")
    df.insert(0, "sample_id", pd.read_csv(d / f"{case}_index.csv").sample_id.to_numpy())
    return df


def leak_stats(e: pd.Series, spice, ml) -> dict:
    return {"n": len(e), "bias": e.median(), "median_err": e.abs().median(), "p95_err": e.abs().quantile(0.95),
            "mean_accuracy_pct": 100 * (1 - e.abs().mean()),
            "r2_log10": float(1 - np.sum((np.log10(spice) - np.log10(ml)) ** 2)
                              / np.sum((np.log10(spice) - np.log10(spice).mean()) ** 2))}


def gate_level(pred: LeakagePredictor, refs: pd.DataFrame):
    cols = KEY + ["vdd", "temp"] + [c for c in refs.columns if c[:2] in ("n_", "p_")] + \
           [c for c in ("v_out", "v_x", "v_y", "v_n1") if c in refs]
    g, _ = predict_gate_records(refs[cols], pred)
    m = g.merge(refs[KEY + ["i_leak"]].rename(columns={"i_leak": "spice"}), on=KEY)
    m["e"] = (m.i_leak - m.spice) / m.spice
    return m


def circuit_level(pred: LeakagePredictor, gate_refs: pd.DataFrame, cla: pd.DataFrame):
    gm = GateLeakageModel(gate_refs, pred)
    ml = predict_circuit_ml(build_cla3(), range(64), sorted(cla.corner_id.unique()), gm.leakage)
    m = cla[["vector_id", "corner_id", "temp", "i_leak"]].rename(columns={"i_leak": "spice"}).merge(
        ml, on=["vector_id", "corner_id"])
    m["e"] = (m.i_leak_ml - m.spice) / m.spice
    return m


def parity(ax, x, y, title):
    ax.scatter(x, y, s=4, alpha=0.5, lw=0)
    lo, hi = min(x.min(), y.min()) / 1.3, max(x.max(), y.max()) * 1.3
    ax.plot([lo, hi], [lo, hi], "k-", lw=0.8)
    for f in (1.05, 1.10):
        ax.plot([lo, hi], [lo * f, hi * f], "k:", lw=0.6); ax.plot([lo, hi], [lo / f, hi / f], "k:", lw=0.6)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_title(title, fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_xlabel("SPICE (A)"); ax.set_ylabel("ML (A)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--refs", choices=["original", "corrected"], default="original",
                    help="corrected = SPICE references with spurious operating points re-solved "
                         "(scripts/d5_op_uniqueness.py); models and everything else unchanged")
    a = ap.parse_args()
    global OUT, RECORD, REFS
    REFS = a.refs
    if a.refs == "corrected":
        OUT = CFG.results / "improve" / "final_corrected_refs"
        RECORD = OUT / "final_eval_record.json"
    if RECORD.exists() and not a.force:
        print(f"Final evaluation already done ({RECORD}); refusing to repeat.")
        return 2
    OUT.mkdir(parents=True, exist_ok=True)
    fe = CFG.data / "final_eval"
    for f in ("gates/gate_spice_results.csv", "circuit/cla3_spice_results.csv"):
        d = pd.read_csv(fe / f)
        assert (d.status == "ok").all(), f"{f}: not all SPICE references valid"
    preds = {k: LeakagePredictor(models_dir=v) for k, v in MODEL_SETS.items()}
    hashes = {k: {c: sha256(v / f"{c}.joblib") for c in CFG.cases} for k, v in MODEL_SETS.items()}

    real = subprocess.run

    def blocked(*x, **y):
        raise RuntimeError("no SPICE during ML evaluation")
    subprocess.run = blocked
    rows, tr_rows, store = [], [], {}
    try:
        # ---------------- transistor level
        for label, dname, primary in (("transistor uniform (new)", "final_eval/transistor_uniform", True),
                                      ("transistor targeted (new)", "final_eval/transistor_targeted", True),
                                      ("transistor uniform (A2b test2, seen)", "transistor_test2_uniform", False),
                                      ("transistor targeted (A2b test2, seen)", "transistor_test2_targeted", False)):
            for case in CFG.cases:
                te = load_tr(CFG.data / dname, case)
                for ms, p in preds.items():
                    met = evaluate(te, p.model(case).predict(te[INPUTS])).set_index("terminal")
                    tr_rows.append({"dataset": label, "primary": primary, "case": case, "models": ms,
                                    **{f"{t}_med": met.loc[t, "med_rel"] for t in TARGETS},
                                    "mean_med": met.loc["mean", "med_rel"], "mean_p95": met.loc["mean", "p95_rel"],
                                    "mean_accuracy_pct": 100 * (1 - met.loc["mean", "mean_rel"])})
        # ---------------- gate + circuit level
        for label, gdir, cdir, primary in (("NEW corners (seed 5005)", fe / "gates", fe / "circuit", True),
                                           ("Part B/C corners (seen)", CFG.data / "gates", CFG.data / "circuit", False)):
            gfile = gdir / ("gate_spice_results_corrected.csv" if REFS == "corrected" else "gate_spice_results.csv")
            cfile = cdir / ("cla3_spice_results_corrected.csv" if REFS == "corrected" else "cla3_spice_results.csv")
            grefs = pd.read_csv(gfile if gfile.exists() else gdir / "gate_spice_results.csv", dtype={"inputs": str})
            cla = pd.read_csv(cfile if cfile.exists() else cdir / "cla3_spice_results.csv")
            for ms, p in preds.items():
                g = gate_level(p, grefs)
                c = circuit_level(p, grefs, cla)
                store[(label, ms)] = (g, c)
                rows.append({"level": "gate (18 cases x 200)", "dataset": label, "primary": primary, "models": ms,
                             **leak_stats(g.e, g.spice, g.i_leak)})
                rows.append({"level": "CLA3 (64 x 200)", "dataset": label, "primary": primary, "models": ms,
                             **leak_stats(c.e, c.spice, c.i_leak_ml)})
    finally:
        subprocess.run = real

    tr = pd.DataFrame(tr_rows); lv = pd.DataFrame(rows)
    tr.to_csv(OUT / "transistor_before_after.csv", index=False)
    lv.to_csv(OUT / "gate_circuit_before_after.csv", index=False)
    # per-case gate table on new corners (after)
    g_new, c_new = store[("NEW corners (seed 5005)", "after")]
    gcase = g_new.groupby(["gate", "inputs"]).apply(lambda d: pd.Series(
        {"spice_median": d.spice.median(), "ml_median": d.i_leak.median(), "bias": d.e.median(),
         "median_err": d.e.abs().median(), "p95_err": d.e.abs().quantile(0.95)}), include_groups=False).reset_index()
    gcase.to_csv(OUT / "gate_18cases_after_new_corners.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    for j, ms in enumerate(("before", "after")):
        g, c = store[("NEW corners (seed 5005)", ms)]
        s1 = leak_stats(g.e, g.spice, g.i_leak); s2 = leak_stats(c.e, c.spice, c.i_leak_ml)
        parity(axes[0, j], g.spice, g.i_leak, f"Gates, NEW corners — {ms}: median {s1['median_err']:.2%}, "
                                              f"p95 {s1['p95_err']:.1%}, acc {s1['mean_accuracy_pct']:.2f} %")
        parity(axes[1, j], c.spice, c.i_leak_ml, f"CLA3, NEW corners — {ms}: median {s2['median_err']:.2%}, "
                                                 f"p95 {s2['p95_err']:.1%}, acc {s2['mean_accuracy_pct']:.2f} %")
    fig.suptitle("Before vs after the improvement phase (independent final-evaluation data)", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / "fig_before_after_parity.png", dpi=140); plt.close(fig)

    rec = {"evaluated": dt.datetime.now().isoformat(timespec="seconds"), "forced": bool(a.force), "refs": REFS, "model_hashes": hashes,
           "final_eval_data_sha256": {f: sha256(fe / f) for f in ("gates/gate_spice_results.csv", "gates/corners.csv",
                                                                   "circuit/cla3_spice_results.csv")}}
    RECORD.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(lv.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(tr.groupby(["dataset", "models"])[["mean_med", "mean_p95", "mean_accuracy_pct"]].mean().to_string(
        float_format=lambda x: f"{x:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
