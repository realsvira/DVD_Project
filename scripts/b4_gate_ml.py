"""Part B steps 4-5: predict gate leakage hierarchically from the four transistor ML
models and compare with the SPICE references at exactly matching corners and inputs.

Method (see src/leakage/gate_ml.py for the full documentation)
  * terminal voltages: inputs exactly 0/VDD; internal and output nodes = SPICE-recorded
    voltages from Part B step 3 (allowed by the PDF at this stage); bodies on rails.
  * ON/OFF model by gate bias (NMOS ON: Vg=VDD; PMOS ON: Vg=0); gates driven by n1 are
    snapped to the nearest rail (snap recorded).
  * I_leak_ML = sum_dev[(Vd-Vs)Id + (Vg-Vs)Ig + (Vb-Vs)Ib] / VDD  (= P_leak/VDD, the same
    definition as the SPICE reference).
  * Inference runs with subprocess execution blocked: no SPICE is used for prediction.

Error definitions per (gate, inputs, corner):  e = (I_ML - I_SPICE) / I_SPICE
  bias = median(e);  median error = median|e|;  p95 error = 95th percentile of |e|;
  mean accuracy = 100 % * (1 - mean|e|).

Outputs
  data/gates/gate_ml_predictions.csv     3,600 rows: ML and SPICE leakage, errors, diagnostics
  data/gates/gate_ml_devices.csv         per transistor: terminal V, model case, ML and SPICE currents
  results/b4/gate_accuracy_18cases.csv/.md, per_gate_summary.csv, error_by_condition.csv,
  results/b4/dominant_error_devices.csv, fig_gate_parity.png, fig_gate_parity_by_gate.png,
  fig_gate_error_by_case.png, b4_report.md (copied to reports/)

Run:  .\\.venv\\Scripts\\python.exe scripts\\b4_gate_ml.py
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from leakage.config import load_config  # noqa: E402
from leakage.gate_ml import device_power, predict_gate_records  # noqa: E402
from leakage.inference import LeakagePredictor  # noqa: E402

CFG = load_config()
GD = CFG.data / "gates"
OUT = CFG.results / "b4"
KEY = ["gate", "inputs", "corner_id"]
GATE_ORDER = ["INV", "NAND2", "NOR2", "AND2", "OR2"]
COLORS = dict(zip(GATE_ORDER, ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]))


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def md(df: pd.DataFrame, fmt="{:.4g}") -> str:
    cols = list(df.columns)
    rows = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        rows.append("| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
                                      for v in r) + " |")
    return "\n".join(rows)


def model_selection_note(overall: pd.Series) -> list[str]:
    """Disclosure if the final transistor models were changed after gate results were seen."""
    amend_f = CFG.results / "a2b" / "selection_amendment.json"
    arch = CFG.results / "b4_archive_off_v1" / "b4_record.json"
    if not amend_f.exists():
        return []
    a = json.loads(amend_f.read_text(encoding="utf-8"))
    lines = ["## Model-selection disclosure", "",
             f"* Change: {a['change']}", f"* Reason (validation only): {a['reason']}",
             f"* Disclosure: {a['disclosure']}"]
    if arch.exists():
        o = json.loads(arch.read_text(encoding="utf-8"))["overall"]
        lines += ["", "| Transistor models | median err | p95 err | bias | mean accuracy |", "|---|---|---|---|---|",
                  f"| v2 ON + v1 OFF (archived) | {o['median_err']:.2%} | {o['p95_err']:.2%} | {o['bias']:+.2%} | "
                  f"{o['mean_accuracy_pct']:.2f} % |",
                  f"| v2 ON + v2 OFF (final, this report) | {overall.median_err:.2%} | {overall.p95_err:.2%} | "
                  f"{overall.bias:+.2%} | {overall.mean_accuracy_pct:.2f} % |"]
    return lines + [""]


def case_stats(g: pd.DataFrame) -> pd.Series:
    e = g.rel_err
    return pd.Series({"n": len(g), "spice_median_A": g.i_leak_spice.median(), "ml_median_A": g.i_leak_ml.median(),
                      "bias": e.median(), "median_err": e.abs().median(), "p95_err": e.abs().quantile(0.95),
                      "max_err": e.abs().max(), "mean_accuracy_pct": 100 * (1 - e.abs().mean()),
                      "frac_within_1pct": (e.abs() < 0.01).mean(), "frac_within_5pct": (e.abs() < 0.05).mean(),
                      "n_extrapolated": int(g.extrapolated.sum())})


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ref = pd.read_csv(GD / "gate_spice_results.csv", dtype={"inputs": str})
    corners = pd.read_csv(GD / "corners.csv")
    chk = ref.merge(corners, on="corner_id", suffixes=("", "_c"))
    assert np.allclose(chk.vdd, chk.vdd_c) and np.allclose(chk.temp, chk.temp_c), "corner mismatch"
    assert (ref.status == "ok").all() and ref.logic_ok.all(), "SPICE references must all be valid"

    predictor = LeakagePredictor()
    models = {c: {"sha256": sha256(CFG.trained_models / f"{c}.joblib"), **predictor.info(c)} for c in CFG.cases}

    # -------------------------------------------------------------- predict with SPICE blocked
    calls = {"n": 0}
    real_run = subprocess.run

    def blocked(*a, **k):
        calls["n"] += 1
        raise RuntimeError("SPICE/subprocess call attempted during ML inference")
    subprocess.run = blocked
    try:
        t0 = time.perf_counter()
        inputs = ref[KEY + ["vdd", "temp"] + [c for c in ref.columns if c[:2] in ("n_", "p_")]
                     + [c for c in ("v_out", "v_x", "v_y", "v_n1") if c in ref]]
        gml, dml = predict_gate_records(inputs, predictor)
        infer_s = time.perf_counter() - t0
    finally:
        subprocess.run = real_run
    print(f"ML inference for {len(gml)} gate cases ({len(dml)} transistor evaluations): "
          f"{infer_s:.2f} s, subprocess calls attempted: {calls['n']}")

    # -------------------------------------------------------------- compare
    res = gml.rename(columns={"i_leak": "i_leak_ml", "p_leak": "p_leak_ml", "i_vdd": "i_vdd_ml",
                              "i_in_a": "i_in_a_ml", "i_in_b": "i_in_b_ml"}).merge(
        ref[KEY + ["temp", "n_vth0", "p_vth0", "i_leak", "i_vdd", "i_in_a", "i_in_b"]]
        .rename(columns={"i_leak": "i_leak_spice", "i_vdd": "i_vdd_spice",
                         "i_in_a": "i_in_a_spice", "i_in_b": "i_in_b_spice"}), on=KEY, validate="one_to_one")
    assert len(res) == len(ref) == 3600
    res["rel_err"] = (res.i_leak_ml - res.i_leak_spice) / res.i_leak_spice
    # diagnostic: direct supply accounting (VDD-rail current + input power), NOT the reported metric
    p_in = (res.vdd * res.i_in_a_ml.where(res.inputs.str[0] == "1", 0.0)
            + res.vdd * res.i_in_b_ml.where(res.inputs.str[-1].where(res.inputs.str.len() > 1) == "1", 0.0))
    res["i_leak_ml_direct"] = res.i_vdd_ml + p_in / res.vdd
    res["rel_err_direct"] = (res.i_leak_ml_direct - res.i_leak_spice) / res.i_leak_spice
    res.to_csv(GD / "gate_ml_predictions.csv", index=False)

    # device level: ML vs SPICE per transistor
    sdev = pd.read_csv(GD / "gate_spice_devices.csv", dtype={"inputs": str})
    dev = dml.merge(sdev[KEY + ["device", "id", "ig", "is", "ib"]].rename(
        columns={"id": "id_spice", "ig": "ig_spice", "is": "is_spice", "ib": "ib_spice"}),
        on=KEY + ["device"], validate="one_to_one")
    dev["p_ml"] = device_power(dev)
    dev["p_spice"] = ((dev.vd - dev.vs) * dev.id_spice + (dev.vg_node - dev.vs) * dev.ig_spice
                      + (dev.vb - dev.vs) * dev.ib_spice)
    dev.to_csv(GD / "gate_ml_devices.csv", index=False)

    # -------------------------------------------------------------- 18-case table
    order = {g: i for i, g in enumerate(GATE_ORDER)}
    tab = res.groupby(["gate", "inputs"]).apply(case_stats, include_groups=False).reset_index()
    tab = tab.sort_values(["gate", "inputs"], key=lambda s: s.map(order) if s.name == "gate" else s)
    tab.to_csv(OUT / "gate_accuracy_18cases.csv", index=False)
    per_gate = res.groupby("gate").apply(case_stats, include_groups=False).reset_index()
    per_gate = per_gate.sort_values("gate", key=lambda s: s.map(order))
    overall = case_stats(res)
    per_gate.to_csv(OUT / "per_gate_summary.csv", index=False)

    # -------------------------------------------------------------- error analysis
    d = dev.copy()
    d["dp"] = d.p_ml - d.p_spice
    tot = d.groupby(KEY).dp.transform(lambda x: x.abs().sum())
    d["share"] = d.dp.abs() / tot.where(tot > 0, 1)
    dom = (d.sort_values("share", ascending=False).groupby(["gate", "inputs"]).apply(
        lambda g: pd.Series({"top_device_by_mean_share": g.groupby(["device", "case"]).share.mean().idxmax()}),
        include_groups=False).reset_index())
    share = d.groupby(["gate", "inputs", "device", "case"]).share.mean().reset_index()
    share.to_csv(OUT / "error_share_by_device.csv", index=False)
    dom.to_csv(OUT / "dominant_error_devices.csv", index=False)
    # device current accuracy (ML vs SPICE, same terminal voltages)
    dacc = []
    for (case, t), _ in [((c, t), None) for c in CFG.cases for t in ("id", "ig", "ib")]:
        x = d[d.case == case]
        if len(x):
            rel = (x[t] - x[f"{t}_spice"]).abs() / np.maximum(x[f"{t}_spice"].abs(), 1e-15)
            dacc.append({"model_case": case, "terminal": t, "n": len(x), "median_rel_err": rel.median(),
                         "p95_rel_err": rel.quantile(0.95)})
    dacc = pd.DataFrame(dacc)
    dacc.to_csv(OUT / "device_current_accuracy_in_gates.csv", index=False)
    cond = []
    for col, bins, labels in (("temp", [-40, 15, 70, 125.01], ["-40..15 C", "15..70 C", "70..125 C"]),
                              ("vdd", [0.9, 0.9667, 1.0333, 1.1001], ["0.90-0.97 V", "0.97-1.03 V", "1.03-1.10 V"])):
        cat = pd.cut(res[col], bins=bins, labels=labels, include_lowest=True)
        for (gname, b), g in res.groupby(["gate", cat], observed=True):
            cond.append({"gate": gname, "by": col, "bin": str(b), "n": len(g), "bias": g.rel_err.median(),
                         "median_err": g.rel_err.abs().median(), "p95_err": g.rel_err.abs().quantile(0.95)})
    cond = pd.DataFrame(cond)
    cond.to_csv(OUT / "error_by_condition.csv", index=False)
    corr = {c: float(res.rel_err.corr(res[c], method="spearman")) for c in ("temp", "vdd", "n_vth0", "p_vth0")}
    worst = res.reindex(res.rel_err.abs().sort_values(ascending=False).index).head(20)[
        KEY + ["temp", "vdd", "i_leak_spice", "i_leak_ml", "rel_err"]]
    worst.to_csv(OUT / "worst_20_cases.csv", index=False)

    # -------------------------------------------------------------- plots
    fig, ax = plt.subplots(figsize=(7.5, 7))
    for gname in GATE_ORDER:
        x = res[res.gate == gname]
        ax.scatter(x.i_leak_spice, x.i_leak_ml, s=7, alpha=0.6, lw=0, c=COLORS[gname], label=f"{gname} (n={len(x)})")
    lo, hi = res[["i_leak_spice", "i_leak_ml"]].min().min() / 1.5, res[["i_leak_spice", "i_leak_ml"]].max().max() * 1.5
    ax.plot([lo, hi], [lo, hi], "k-", lw=0.8, label="ML = SPICE")
    for f, ls in ((1.05, "--"), (1.10, ":")):
        ax.plot([lo, hi], [lo * f, hi * f], "k" + ls, lw=0.6)
        ax.plot([lo, hi], [lo / f, hi / f], "k" + ls, lw=0.6, label=f"±{(f - 1) * 100:.0f} %")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("SPICE gate leakage I_leak (A)"); ax.set_ylabel("ML (hierarchical) gate leakage (A)")
    ax.set_title(f"Gate leakage: ML vs SPICE, 18 cases x 200 corners\nmedian |err| {overall.median_err:.2%}, "
                 f"p95 {overall.p95_err:.2%}, mean accuracy {overall.mean_accuracy_pct:.2f} %", fontsize=11)
    ax.grid(True, which="major", alpha=0.3); ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(OUT / "fig_gate_parity.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.6))
    for ax, gname in zip(axes, GATE_ORDER):
        x = res[res.gate == gname]
        for k, bits in enumerate(sorted(x.inputs.unique())):
            y = x[x.inputs == bits]
            ax.scatter(y.i_leak_spice, y.i_leak_ml, s=6, alpha=0.6, lw=0, label=bits, c=f"C{k}")
        l2, h2 = x[["i_leak_spice", "i_leak_ml"]].min().min() / 1.5, x[["i_leak_spice", "i_leak_ml"]].max().max() * 1.5
        ax.plot([l2, h2], [l2, h2], "k-", lw=0.8)
        ax.plot([l2, h2], [l2 * 1.1, h2 * 1.1], "k:", lw=0.6); ax.plot([l2, h2], [l2 / 1.1, h2 / 1.1], "k:", lw=0.6)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(l2, h2); ax.set_ylim(l2, h2)
        s = per_gate.set_index("gate").loc[gname]
        ax.set_title(f"{gname}: median {s.median_err:.2%}, p95 {s.p95_err:.1%}", fontsize=10)
        ax.set_xlabel("SPICE (A)"); ax.set_ylabel("ML (A)"); ax.grid(True, alpha=0.3)
        ax.legend(title="inputs", fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "fig_gate_parity_by_gate.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.2))
    labels = [f"{g}\n{b}" for g, b in zip(tab.gate, tab.inputs)]
    data = [res[(res.gate == g) & (res.inputs == b)].rel_err * 100 for g, b in zip(tab.gate, tab.inputs)]
    ax.boxplot(data, tick_labels=labels, showfliers=True, flierprops={"markersize": 2})
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("signed error (ML − SPICE)/SPICE  (%)")
    ax.set_title("Gate leakage error distribution per gate/input case (200 corners each)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "fig_gate_error_by_case.png", dpi=140); plt.close(fig)

    # -------------------------------------------------------------- report
    best = tab.nsmallest(3, "median_err")[["gate", "inputs", "median_err", "p95_err"]]
    worst_c = tab.nlargest(3, "median_err")[["gate", "inputs", "median_err", "p95_err"]]
    show = tab[["gate", "inputs", "spice_median_A", "ml_median_A", "bias", "median_err", "p95_err",
                "mean_accuracy_pct", "frac_within_5pct"]]
    pct = show.copy()
    for c in ("bias", "median_err", "p95_err", "frac_within_5pct"):
        pct[c] = pct[c].map(lambda v: f"{v:+.2%}" if c == "bias" else f"{v:.2%}")
    pct["mean_accuracy_pct"] = pct.mean_accuracy_pct.map(lambda v: f"{v:.2f}")
    record = {"created": dt.datetime.now().isoformat(timespec="seconds"), "models": models,
              "inference_seconds": round(infer_s, 2), "subprocess_calls_during_inference": calls["n"],
              "overall": overall.to_dict(), "spearman_signed_error_vs": corr,
              "direct_accounting_median_err": float(res.rel_err_direct.abs().median()),
              "direct_accounting_p95_err": float(res.rel_err_direct.abs().quantile(0.95))}
    (OUT / "b4_record.json").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    tab.to_markdown if False else None
    rep = ["# Part B steps 4–5: hierarchical ML gate leakage vs SPICE", "",
           f"Generated {record['created']}. 3,600 predictions (18 gate/input cases × 200 corners) from the four "
           "transistor ML models (" + ", ".join(f"{c}: {m['algorithm']} {m.get('transform', '')}" for c, m in models.items())
           + f"). Inference took {infer_s:.1f} s with subprocess execution blocked: "
           f"{calls['n']} SPICE/subprocess calls were attempted.", "",
           "**Dependency on SPICE node voltages (allowed by the PDF at this stage):** the internal-node and "
           "output voltages (out, x, y, n1) used to set each transistor's terminal voltages are the "
           "SPICE-recorded values from Part B step 3. Inputs are exactly 0/VDD and bodies sit on the rails. "
           "Only the terminal *currents* come from the ML models; the predictions are therefore not "
           "independent of SPICE-generated node voltages.", "",
           *model_selection_note(overall),
           "## Method", "",
           "* ON/OFF model per transistor by gate bias (NMOS ON: Vg = VDD; PMOS ON: Vg = 0). AND2/OR2 "
           f"inverters are driven by n1, a few µV from a rail (max snap {res.vg_snap.max():.2e} V); they are "
           "classified by the nearest rail and the model input is snapped to it. A gate-biased-ON transistor "
           "whose source sits at an intermediate node (e.g. NAND2 '10', MNA with source at x ≈ 0.77 V) still "
           "uses the ON model, whose training data covers sources anywhere in [0, VDD].",
           "* Leakage accounting: I_leak = Σ_transistors [(Vd−Vs)·Id + (Vg−Vs)·Ig + (Vb−Vs)·Ib] / VDD, i.e. total "
           "device power / VDD. This equals the SPICE definition (power delivered by the VDD rail and the "
           "input sources, / VDD) by Tellegen's theorem. Using the source as reference enforces KCL per device "
           "and counts every terminal once; an ON transistor's channel current only enters through its tiny "
           "Vds. Verified: the same code fed with SPICE's own per-transistor currents reproduces the SPICE "
           "leakage of all 3,600 cases (`tests/test_gate_ml.py`). VDD-rail and input-source currents are "
           "included through the total power; they are also computed directly as diagnostics.",
           "* Error per sample e = (I_ML − I_SPICE)/I_SPICE; bias = median(e); median error = median|e|; "
           "p95 error = 95th percentile of |e|; mean accuracy = 100 %·(1 − mean|e|).", "",
           "## Results: all 18 gate/input cases", "", md(pct), "",
           f"**Overall (3,600):** median |err| {overall.median_err:.2%}, p95 {overall.p95_err:.2%}, "
           f"bias {overall.bias:+.2%}, mean accuracy **{overall.mean_accuracy_pct:.2f} %**, "
           f"{overall.frac_within_5pct:.1%} of predictions within ±5 %.", "",
           "![parity](fig_gate_parity.png)", "", "![parity by gate](fig_gate_parity_by_gate.png)", "",
           "![error by case](fig_gate_error_by_case.png)", "",
           "## Per gate", "", md(per_gate[["gate", "n", "bias", "median_err", "p95_err", "mean_accuracy_pct"]]), "",
           "**Most accurate cases:** " + "; ".join(f"{r.gate} {r.inputs} (median {r.median_err:.2%}, p95 {r.p95_err:.1%})"
                                                     for r in best.itertuples()) + ".", "",
           "**Least accurate cases:** " + "; ".join(f"{r.gate} {r.inputs} (median {r.median_err:.2%}, p95 {r.p95_err:.1%})"
                                                      for r in worst_c.itertuples()) + ".", "",
           "## Error analysis", "",
           "Transistor-level accuracy inside the gates (ML vs SPICE currents at identical terminal voltages):", "",
           md(dacc), "",
           "Spearman correlation of the signed gate error with: " + ", ".join(f"{k} {v:+.2f}" for k, v in corr.items())
           + ". Error by temperature and VDD bin: `error_by_condition.csv`.", "",
           "Which transistor dominates the gate error (mean share of |ΔP|):", "", md(dom), "",
           f"Diagnostic, not the reported metric: computing leakage directly from the predicted VDD-rail and "
           f"input currents gives median |err| {record['direct_accounting_median_err']:.2%} and p95 "
           f"{record['direct_accounting_p95_err']:.1%}; it depends on the ON transistors' channel currents at "
           "Vds ≈ 0 and on per-device KCL, which the power formulation avoids.", "",
           "Worst 20 samples: `worst_20_cases.csv`. Data: `data/gates/gate_ml_predictions.csv`, "
           "`data/gates/gate_ml_devices.csv`.", ""]
    (OUT / "b4_report.md").write_text("\n".join(rep), encoding="utf-8")
    (CFG.reports / "b4_gate_ml_report.md").write_text("\n".join(rep).replace("](fig_", "](../results/b4/fig_"),
                                                      encoding="utf-8")
    pd.set_option("display.width", 220)
    print(show.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print("\nper gate:\n" + per_gate[["gate", "bias", "median_err", "p95_err", "mean_accuracy_pct"]].to_string(index=False))
    print(f"\nOVERALL median {overall.median_err:.4f}  p95 {overall.p95_err:.4f}  bias {overall.bias:+.4f}  "
          f"mean accuracy {overall.mean_accuracy_pct:.3f} %   extrapolated {int(res.extrapolated.sum())}")
    print("direct-accounting diagnostic: median", round(record["direct_accounting_median_err"], 4),
          "p95", round(record["direct_accounting_p95_err"], 4))
    print("device current accuracy:\n" + dacc.to_string(index=False))
    print("dominant error devices:\n" + dom.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
