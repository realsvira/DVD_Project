"""Final deliverables for the FINAL transistor models (trained_models/<case>.joblib = v3).

Pure evaluation/presentation of data that was already evaluated in the improvement phase
(no model is trained, tuned or selected here). Figures are saved at 200 dpi.

  results/final/transistor_test_results.csv/.md     per case x terminal: R2_t, median/p95 rel. error,
                                                    abs. error, sign accuracy (independent test set,
                                                    data/final_eval/transistor_uniform, + targeted)
  results/final/fig_transistor_pred_vs_spice_<case>.png   (4 plots, deliverable 5)
  results/final/gate_predictions_<set>.csv          every gate record: SPICE and ML leakage, the SPICE
                                                    node voltages (spice_node_*) and the node voltages
                                                    given to the transistor models (ml_input_node_*;
                                                    the same SPICE values, read back from the model
                                                    inputs -- not an independent ML prediction)
  results/final/gate_18cases_partB.csv/.md          deliverable 6 on the Part B corners (spec set)
  results/final/gate_18cases_newcorners.csv         same on the independent new corners
  results/final/fig_gate_parity.png                 deliverable 7 (both corner sets, log axes)
  results/final/cla3_results.csv/.md                deliverable 8 (both corner sets, all 128 input
                                                    combinations A, B, Cin; plus Cin = 0 / 1 halves)
  results/final/fig_cla3_parity.png                 deliverable 8 parity (log axes)
  results/final/summary.json

Run:  .venv/bin/python scripts/z1_final_deliverables.py
"""
from __future__ import annotations

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
sys.path.insert(0, str(ROOT / "scripts"))
from a2_evaluate_test import plot_case  # noqa: E402
from leakage.circuit import build_cla3, predict_circuit_ml  # noqa: E402
from leakage.config import load_config  # noqa: E402
from leakage.gate_ml import GateLeakageModel, predict_gate_records  # noqa: E402
from leakage.inference import LeakagePredictor  # noqa: E402
from leakage.ml import INPUTS, evaluate  # noqa: E402

CFG = load_config()
OUT = CFG.results / "final"
KEY = ["gate", "inputs", "corner_id"]
ORDER = ["INV", "NAND2", "NOR2", "AND2", "OR2"]


def md(df, fmt="{:.4g}"):
    cols = list(df.columns)
    rows = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        rows.append("| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(rows)


def load_tr(d, case):
    df = pd.read_csv(d / f"{case}.csv")
    df.insert(0, "sample_id", pd.read_csv(d / f"{case}_index.csv").sample_id.to_numpy())
    return df


def stats(e, spice, ml):
    return {"n": len(e), "spice_median_A": float(np.median(spice)), "ml_median_A": float(np.median(ml)),
            "bias": float(np.median(e)), "median_err": float(np.median(np.abs(e))),
            "p95_err": float(np.quantile(np.abs(e), 0.95)), "max_err": float(np.max(np.abs(e))),
            "mean_accuracy_pct": float(100 * (1 - np.mean(np.abs(e)))),
            "r2_log10": float(1 - np.sum((np.log10(spice) - np.log10(ml)) ** 2)
                              / np.sum((np.log10(spice) - np.mean(np.log10(spice))) ** 2))}


NODES = ("out", "x", "y", "n1")


def ml_input_nodes(dev: pd.DataFrame) -> pd.DataFrame:
    """Voltage of every gate node as it was given to the transistor models (read back from the
    per-transistor model inputs: drain/source voltages and the actual gate-node voltage)."""
    parts = []
    for term, col in (("d", "vd"), ("s", "vs"), ("g", "vg_node")):
        t = dev[KEY + [f"node_{term}", col]].rename(columns={f"node_{term}": "node", col: "v"})
        parts.append(t[t.node.isin(NODES)])
    t = pd.concat(parts, ignore_index=True)
    spread = t.groupby(KEY + ["node"]).v.agg(lambda v: v.max() - v.min())
    assert float(spread.max()) == 0.0, "a node was given different voltages within one record"
    w = t.groupby(KEY + ["node"]).v.first().unstack("node")
    return w.rename(columns={n: f"ml_input_node_{n}" for n in w.columns}).reset_index()


def gate_eval(pred, refs):
    cols = KEY + ["vdd", "temp"] + [c for c in refs.columns if c[:2] in ("n_", "p_")] + \
           [c for c in ("v_out", "v_x", "v_y", "v_n1") if c in refs]
    g, dev = predict_gate_records(refs[cols], pred)
    m = g.merge(refs[KEY + ["i_leak"]].rename(columns={"i_leak": "spice"}), on=KEY)
    m["e"] = (m.i_leak - m.spice) / m.spice
    spice_nodes = refs[KEY + [f"v_{n}" for n in NODES if f"v_{n}" in refs]].rename(
        columns={f"v_{n}": f"spice_node_{n}" for n in NODES})
    m = m.merge(spice_nodes, on=KEY).merge(ml_input_nodes(dev), on=KEY, how="left")
    for n in NODES:                                   # ML inputs are the SPICE values (spec step 4)
        if f"ml_input_node_{n}" in m:
            both = m[f"ml_input_node_{n}"].notna()
            assert np.array_equal(m.loc[both, f"ml_input_node_{n}"].to_numpy(),
                                  m.loc[both, f"spice_node_{n}"].to_numpy()), n
    pairs = [c for n in NODES for c in (f"spice_node_{n}", f"ml_input_node_{n}") if c in m]
    return m[[c for c in m.columns if c not in pairs] + pairs]


def load_cla_refs(cd: Path) -> pd.DataFrame:
    """CLA3 SPICE references, all available input combinations (Cin = 0 file + Cin = 1 file)."""
    parts = [pd.read_csv(cd / "cla3_spice_results.csv")]
    if (cd / "cla3_spice_results_cin1.csv").exists():
        parts.append(pd.read_csv(cd / "cla3_spice_results_cin1.csv"))
    cla = pd.concat(parts, ignore_index=True)
    assert not cla.duplicated(["vector_id", "corner_id"]).any()
    assert (cla.status == "ok").all() and cla.sum_ok.all() and cla.all_nodes_logic_ok.all()
    return cla


def gate_table(m):
    t = m.groupby(["gate", "inputs"]).apply(lambda d: pd.Series(stats(d.e.to_numpy(), d.spice.to_numpy(),
                                                                      d.i_leak.to_numpy())), include_groups=False)
    t = t.reset_index()
    t["gate"] = pd.Categorical(t.gate, ORDER, ordered=True)
    return t.sort_values(["gate", "inputs"]).reset_index(drop=True)


def parity(ax, x, y, c, title, cmap=None, label=None):
    sc = ax.scatter(x, y, c=c, s=5, alpha=0.6, lw=0, cmap=cmap, label=label)
    lo, hi = min(np.min(x), np.min(y)) / 1.3, max(np.max(x), np.max(y)) * 1.3
    ax.plot([lo, hi], [lo, hi], "k-", lw=0.8)
    for f, ls in ((1.05, "--"), (1.10, ":")):
        ax.plot([lo, hi], [lo * f, hi * f], "k" + ls, lw=0.6); ax.plot([lo, hi], [lo / f, hi / f], "k" + ls, lw=0.6)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_title(title, fontsize=10); ax.grid(True, alpha=0.3)
    return sc


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pred = LeakagePredictor()                       # final models = trained_models/<case>.joblib (v3)
    real = subprocess.run

    def blocked(*a, **k):
        raise RuntimeError("no SPICE during ML evaluation")
    subprocess.run = blocked
    summary = {"models": {c: pred.info(c) for c in CFG.cases}}
    try:
        # ---------------- transistor level (deliverable 5)
        rows = []
        for label, d in (("independent test (spec distribution)", CFG.data / "final_eval" / "transistor_uniform"),
                         ("independent test (near-rail / small |Vds|)", CFG.data / "final_eval" / "transistor_targeted")):
            for case in CFG.cases:
                te = load_tr(d, case)
                p = pred.model(case).predict(te[INPUTS])
                met = evaluate(te, p)
                met.insert(0, "case", case); met.insert(0, "test_set", label)
                rows.append(met)
                if "spec" in label:
                    plot_case(f"{case} — final model, independent test", te, p,
                              met.drop(columns=["case", "test_set"]), OUT / f"fig_transistor_pred_vs_spice_{case}.png")
        tr = pd.concat(rows, ignore_index=True)
        tr.to_csv(OUT / "transistor_test_results.csv", index=False)
        cols = ["test_set", "case", "terminal", "r2_t", "med_rel", "p95_rel", "frac_rel_lt_1pct", "med_abs", "sign_acc"]
        (OUT / "transistor_test_results.md").write_text(md(tr[cols]), encoding="utf-8")

        # ---------------- gates (deliverables 6, 7) and CLA (deliverable 8)
        sets = {"Part B/C corners (seed 4004)": (CFG.data / "gates", CFG.data / "circuit"),
                "independent new corners (seed 5005)": (CFG.data / "final_eval" / "gates", CFG.data / "final_eval" / "circuit")}
        gres, cres = {}, {}
        for label, (gd, cd) in sets.items():
            gf = gd / "gate_spice_results_corrected.csv"
            refs = pd.read_csv(gf if gf.exists() else gd / "gate_spice_results.csv", dtype={"inputs": str})
            m = gate_eval(pred, refs)
            gres[label] = m
            cla = load_cla_refs(cd)
            vids = sorted(cla.vector_id.unique())
            cids = sorted(cla.corner_id.unique())
            gm = GateLeakageModel(refs, pred)
            ml = predict_circuit_ml(build_cla3(), vids, cids, gm.leakage)
            ref_i = refs.set_index(["gate", "inputs", "corner_id"]).i_leak

            def spice_gate(kind, bits, corners):     # composition check: isolated-gate SPICE, no ML
                return pd.Series([ref_i.loc[(kind, bits, c)] for c in corners], index=corners)
            comp = predict_circuit_ml(build_cla3(), vids, cids, spice_gate).rename(
                columns={"i_leak_ml": "i_leak_gatesum_spice"})
            cm = cla[["vector_id", "cin", "corner_id", "temp", "i_leak"]].rename(columns={"i_leak": "spice"}).merge(
                ml, on=["vector_id", "corner_id"]).merge(comp, on=["vector_id", "corner_id"])
            assert len(cm) == len(cla)
            cm["e"] = (cm.i_leak_ml - cm.spice) / cm.spice
            cm["e_composition"] = (cm.i_leak_gatesum_spice - cm.spice) / cm.spice
            cres[label] = cm
    finally:
        subprocess.run = real

    for label, tag in (("Part B/C corners (seed 4004)", "partB"), ("independent new corners (seed 5005)", "newcorners")):
        gres[label].drop(columns=[c for c in ("n_devices",) if c in gres[label]]).to_csv(
            OUT / f"gate_predictions_{tag}.csv", index=False)
        cres[label].to_csv(OUT / f"cla3_predictions_{tag}.csv", index=False)
    gb = gate_table(gres["Part B/C corners (seed 4004)"])
    gn = gate_table(gres["independent new corners (seed 5005)"])
    gb.to_csv(OUT / "gate_18cases_partB.csv", index=False)
    gn.to_csv(OUT / "gate_18cases_newcorners.csv", index=False)
    show = gb[["gate", "inputs", "spice_median_A", "ml_median_A", "bias", "median_err", "p95_err"]].copy()
    for c in ("bias",):
        show[c] = show[c].map(lambda v: f"{v:+.2%}")
    for c in ("median_err", "p95_err"):
        show[c] = show[c].map(lambda v: f"{v:.2%}")
    (OUT / "gate_18cases_partB.md").write_text(md(show), encoding="utf-8")

    crow, cin_rows = [], []
    for label, cm in cres.items():
        crow.append({"corner_set": label, "vectors": int(cm.vector_id.nunique()),
                     **stats(cm.e.to_numpy(), cm.spice.to_numpy(), cm.i_leak_ml.to_numpy())})
        for cin, d in cm.groupby("cin"):
            cin_rows.append({"corner_set": label, "subset": f"Cin = {cin}", "vectors": int(d.vector_id.nunique()),
                             **stats(d.e.to_numpy(), d.spice.to_numpy(), d.i_leak_ml.to_numpy())})
        e = cm.e_composition.abs()
        summary.setdefault("cla3_composition", {})[label] = {
            "median_err": float(e.median()), "p95_err": float(e.quantile(0.95)), "max_err": float(e.max())}
    summary["cla3_by_cin"] = cin_rows
    for label, m in gres.items():
        summary.setdefault("gate_overall", {})[label] = stats(m.e.to_numpy(), m.spice.to_numpy(), m.i_leak.to_numpy())
    ct = pd.DataFrame(crow)
    ct.to_csv(OUT / "cla3_results.csv", index=False)
    pd.DataFrame(cin_rows).to_csv(OUT / "cla3_results_by_cin.csv", index=False)
    (OUT / "cla3_results.md").write_text(md(ct) + "\n\nBy carry-in:\n\n" + md(pd.DataFrame(cin_rows)),
                                         encoding="utf-8")
    summary["cla3"] = {r["corner_set"]: {k: v for k, v in r.items() if k != "corner_set"} for r in crow}

    colors = dict(zip(ORDER, ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.6))
    for ax, (label, m) in zip(axes, gres.items()):
        for gname in ORDER:                          # scatter every gate first ...
            x = m[m.gate == gname]
            ax.scatter(x.spice, x.i_leak, c=colors[gname], s=5, alpha=0.6, lw=0, label=f"{gname} (n={len(x)})")
        lo = min(m.spice.min(), m.i_leak.min()) / 1.3  # ... then one set of limits / reference lines
        hi = max(m.spice.max(), m.i_leak.max()) * 1.3
        ax.plot([lo, hi], [lo, hi], "k-", lw=0.8, label="ML = SPICE")
        for f, ls in ((1.05, "--"), (1.10, ":")):
            ax.plot([lo, hi], [lo * f, hi * f], "k" + ls, lw=0.6)
            ax.plot([lo, hi], [lo / f, hi / f], "k" + ls, lw=0.6, label=f"±{(f - 1) * 100:.0f} %")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.grid(True, alpha=0.3)
        s = summary["gate_overall"][label]
        ax.set_title(f"Gates, {label}\nmedian |err| {s['median_err']:.2%}, p95 {s['p95_err']:.2%}, "
                     f"mean accuracy {s['mean_accuracy_pct']:.2f} %", fontsize=10)
        ax.set_xlabel("SPICE gate leakage (A)"); ax.set_ylabel("ML gate leakage (A)"); ax.legend(fontsize=8)
    fig.suptitle("Gate leakage: hierarchical ML (final models) vs SPICE, 18 cases × 200 corners", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / "fig_gate_parity.png", dpi=200); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4))
    for ax, (label, cm) in zip(axes, cres.items()):
        sc = parity(ax, cm.spice.to_numpy(), cm.i_leak_ml.to_numpy(), cm.temp.to_numpy(), "", cmap="coolwarm")
        s = summary["cla3"][label]
        ax.set_title(f"3-bit CLA, {label}\nmedian |err| {s['median_err']:.2%}, p95 {s['p95_err']:.2%}, "
                     f"mean accuracy {s['mean_accuracy_pct']:.2f} %", fontsize=10)
        ax.set_xlabel("full-circuit SPICE leakage (A)"); ax.set_ylabel("hierarchical ML leakage (A)")
        fig.colorbar(sc, ax=ax, label="T (°C)")
    nv = sorted({int(cm.vector_id.nunique()) for cm in cres.values()})
    fig.suptitle(f"3-bit carry-lookahead adder: {'/'.join(map(str, nv))} input combinations (A, B, Cin) "
                 "× 200 corners", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / "fig_cla3_parity.png", dpi=200); plt.close(fig)

    tm = tr[tr.terminal == "mean"].groupby("test_set")[["med_rel", "p95_rel", "mean_rel"]].mean()
    summary["transistor_mean"] = {k: {"median_err": float(r.med_rel), "p95_err": float(r.p95_rel),
                                      "mean_accuracy_pct": float(100 * (1 - r.mean_rel))} for k, r in tm.iterrows()}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("transistor_mean", "gate_overall", "cla3", "cla3_by_cin",
                                              "cla3_composition")}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
