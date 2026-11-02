"""Part C: hierarchical ML leakage of the 3-bit CLA and comparison with full-circuit SPICE.

ML estimate per (input vector, corner)
  1. Boolean simulation of the 36-gate netlist gives every gate's input bits.
  2. Each gate's leakage = Part B gate model (leakage.gate_ml.GateLeakageModel): the four
     transistor ML models applied to that gate type / input pattern / corner, with the
     gate-internal node voltages looked up from the Part B ISOLATED-GATE SPICE references
     (as in the PDF's gate model). This lookup is SPICE-derived (single gates driven by ideal
     0/VDD sources); it is not independent of SPICE.
  3. Circuit leakage = sum of the 36 gate leakages.
  NOTHING from the full-circuit SPICE run (its leakage or its node voltages) is used; the
  ML function only receives (vector_id, corner_id). It runs with subprocess blocked.

Also computed (diagnostic): the same sum using the Part B *SPICE* gate leakages, which
separates the composition error (loading / non-ideal inputs between gates) from the ML error.

Metrics, e = (I_ML - I_SPICE) / I_SPICE over 64 vectors x 200 corners:
  median leakage (SPICE, ML), bias = median(e), median error = median|e|,
  p95 error = 95th percentile |e|, mean accuracy = 100 % * (1 - mean|e|),
  R2 on log10(I) and on I.

Outputs: data/circuit/cla3_ml_predictions.csv, results/c/cla3_evaluation.csv/.md,
results/c/fig_cla3_parity.png, results/c/fig_cla3_error_by_vector.png, reports/c_cla3_report.md

Run:  .\\.venv\\Scripts\\python.exe scripts\\c2_cla_ml.py
"""
from __future__ import annotations

import datetime as dt
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
from leakage.circuit import build_cla3, gate_patterns, predict_circuit_ml, vector_bits  # noqa: E402
from leakage.config import load_config  # noqa: E402
from leakage.gate_ml import GateLeakageModel  # noqa: E402

CFG = load_config()
OUT = CFG.results / "c"
KEY = ["vector_id", "corner_id"]


def r2(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(1 - np.sum((a - b) ** 2) / np.sum((a - a.mean()) ** 2))


def stats(e: pd.Series, spice: pd.Series, ml: pd.Series) -> dict:
    return {"n": len(e), "spice_median_A": spice.median(), "ml_median_A": ml.median(), "bias": e.median(),
            "median_err": e.abs().median(), "p95_err": e.abs().quantile(0.95), "max_err": e.abs().max(),
            "mean_accuracy_pct": 100 * (1 - e.abs().mean()), "frac_within_5pct": (e.abs() < 0.05).mean(),
            "r2_log10": r2(np.log10(spice), np.log10(ml)), "r2_linear": r2(spice, ml)}


def md(df, fmt="{:.4g}"):
    cols = list(df.columns)
    rows = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        rows.append("| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    circ = build_cla3()
    spice = pd.read_csv(CFG.data / "circuit" / "cla3_spice_results.csv")
    assert (spice.status == "ok").all() and spice.sum_ok.all(), "full-circuit SPICE references must be valid"
    corners = sorted(spice.corner_id.unique())
    gate_refs = pd.read_csv(CFG.data / "gates" / "gate_spice_results.csv", dtype={"inputs": str})

    # ---------------------------------------------------------------- ML (SPICE blocked)
    gm = GateLeakageModel(gate_refs)
    calls = {"n": 0}
    real = subprocess.run

    def blocked(*a, **k):
        calls["n"] += 1
        raise RuntimeError("no subprocess / SPICE allowed during ML prediction")
    subprocess.run = blocked
    try:
        t0 = time.perf_counter()
        ml = predict_circuit_ml(circ, range(64), corners, gm.leakage)
        t_ml = time.perf_counter() - t0
    finally:
        subprocess.run = real
    print(f"hierarchical ML for 64 x {len(corners)}: {t_ml:.2f} s, subprocess calls attempted: {calls['n']}")

    # diagnostic: same composition with the Part B SPICE gate leakages
    gs = gate_refs.set_index(["gate", "inputs", "corner_id"]).i_leak

    def spice_gate(kind, bits, cids):
        return pd.Series(gs.loc[[(kind, bits, c) for c in cids]].to_numpy(), index=cids)
    comp = predict_circuit_ml(circ, range(64), corners, spice_gate).rename(columns={"i_leak_ml": "i_leak_gatesum_spice"})

    df = spice[KEY + ["A", "B", "vdd", "temp", "n_vth0", "p_vth0", "i_leak"]].rename(columns={"i_leak": "i_leak_spice"})
    df = df.merge(ml, on=KEY, validate="one_to_one").merge(comp, on=KEY, validate="one_to_one")
    assert len(df) == 64 * len(corners)
    df["rel_err"] = (df.i_leak_ml - df.i_leak_spice) / df.i_leak_spice
    df["composition_err"] = (df.i_leak_gatesum_spice - df.i_leak_spice) / df.i_leak_spice
    df["gate_ml_err"] = (df.i_leak_ml - df.i_leak_gatesum_spice) / df.i_leak_gatesum_spice
    df.to_csv(CFG.data / "circuit" / "cla3_ml_predictions.csv", index=False)

    main_s = stats(df.rel_err, df.i_leak_spice, df.i_leak_ml)
    comp_s = stats(df.composition_err, df.i_leak_spice, df.i_leak_gatesum_spice)
    gml_s = stats(df.gate_ml_err, df.i_leak_gatesum_spice, df.i_leak_ml)
    ev = pd.DataFrame([{"comparison": "hierarchical ML vs full-circuit SPICE (REPORTED)", **main_s},
                       {"comparison": "diag: sum of Part B SPICE gate leakages vs full-circuit SPICE", **comp_s},
                       {"comparison": "diag: hierarchical ML vs sum of SPICE gate leakages", **gml_s}])
    ev.to_csv(OUT / "cla3_evaluation.csv", index=False)
    per_vec = df.groupby(["vector_id", "A", "B"]).apply(
        lambda g: pd.Series(stats(g.rel_err, g.i_leak_spice, g.i_leak_ml)), include_groups=False).reset_index()
    per_vec.to_csv(OUT / "cla3_per_vector.csv", index=False)
    corr = {c: float(df.rel_err.corr(df[c], method="spearman")) for c in ("temp", "vdd", "n_vth0", "p_vth0")}
    # gate-type composition of the circuit for the report
    kinds = pd.Series([g.kind for g in circ.gates]).value_counts().to_dict()
    pats = pd.DataFrame([{"vector_id": v, "kind": k, "bits": b} for v in range(64)
                         for _, k, b in gate_patterns(circ, vector_bits(v))])
    usage = pats.groupby(["kind", "bits"]).size().rename("instances_over_64_vectors").reset_index()
    usage.to_csv(OUT / "cla3_gate_case_usage.csv", index=False)

    # ---------------------------------------------------------------- plots
    fig, ax = plt.subplots(figsize=(7.5, 7))
    sc = ax.scatter(df.i_leak_spice, df.i_leak_ml, c=df.temp, cmap="coolwarm", s=5, alpha=0.6, lw=0)
    lo = min(df.i_leak_spice.min(), df.i_leak_ml.min()) / 1.3
    hi = max(df.i_leak_spice.max(), df.i_leak_ml.max()) * 1.3
    ax.plot([lo, hi], [lo, hi], "k-", lw=0.8, label="ML = SPICE")
    for f, ls in ((1.05, "--"), (1.10, ":")):
        ax.plot([lo, hi], [lo * f, hi * f], "k" + ls, lw=0.6)
        ax.plot([lo, hi], [lo / f, hi / f], "k" + ls, lw=0.6, label=f"±{(f - 1) * 100:.0f} %")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    fig.colorbar(sc, ax=ax, label="T (°C)")
    ax.set_xlabel("full-circuit SPICE leakage (A)"); ax.set_ylabel("hierarchical ML leakage, Σ 36 gates (A)")
    ax.set_title(f"3-bit CLA: ML vs SPICE, 64 vectors × {len(corners)} corners\n"
                 f"median |err| {main_s['median_err']:.2%}, p95 {main_s['p95_err']:.2%}, "
                 f"mean accuracy {main_s['mean_accuracy_pct']:.2f} %", fontsize=11)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(OUT / "fig_cla3_parity.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(per_vec.vector_id, per_vec.median_err * 100, "o-", ms=3, label="median |err|")
    ax.plot(per_vec.vector_id, per_vec.p95_err * 100, "s-", ms=3, label="p95 |err|")
    ax.plot(per_vec.vector_id, per_vec.bias * 100, "^-", ms=3, label="bias")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("input vector id (A + 8·B)"); ax.set_ylabel("error over 200 corners (%)")
    ax.set_title("3-bit CLA: hierarchical ML error per input vector"); ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "fig_cla3_error_by_vector.png", dpi=140); plt.close(fig)

    # ---------------------------------------------------------------- report
    gate_overall = json.loads((CFG.results / "b4" / "b4_record.json").read_text(encoding="utf-8"))["overall"]
    worst_v = per_vec.nlargest(3, "median_err"); best_v = per_vec.nsmallest(3, "median_err")
    show = ev.copy()
    for c in ("bias", "median_err", "p95_err", "max_err", "frac_within_5pct"):
        show[c] = show[c].map(lambda v: f"{v:+.2%}" if c == "bias" else f"{v:.2%}")
    show["mean_accuracy_pct"] = show.mean_accuracy_pct.map(lambda v: f"{v:.2f}")
    rep = ["# Part C: 3-bit carry-lookahead adder — hierarchical ML vs full-circuit SPICE", "",
           f"Generated {dt.datetime.now().isoformat(timespec='seconds')}.", "",
           "## Circuit", "",
           "Lookahead logic with C0 = Cin: Gi = Ai·Bi, Pi = Ai ⊕ Bi, C1 = G0 + P0·C0, C2 = G1 + P1·G0 + P1·P0·C0, "
           "C3 = G2 + P2·G1 + P2·P1·G0 + P2·P1·P0·C0 (Cout), Si = Pi ⊕ Ci. Only INV / NAND2 / AND2 / OR2 are "
           "needed (NOR2 unused); XOR(x,y) = AND2(OR2(x,y), NAND2(x,y)) with NAND2(Ai,Bi) shared by Gi = INV(·) and Pi. "
           f"{len(circ.gates)} gates ({', '.join(f'{v} {k}' for k, v in kinds.items())}), 192 transistors, same "
           "widths / bodies / model cards / corner process draws as Part B. Gate list: "
           "`data/circuit/cla3_gate_list.csv`; complete netlist: `netlists/circuit/CLA3.cir`.", "",
           "**Carry-in assumption:** Cin tied to logic 0 (0 V source); all 6 operand bits exercised → 64 vectors. "
           "The Boolean model is verified exhaustively for Cin = 0 and 1 (128 cases, `tests/test_circuit.py`); "
           "SPICE confirms A+B at the outputs and every internal net at its logic level for all 12,800 runs.", "",
           "## Method", "",
           "* SPICE reference: complete transistor netlist, I_leak = Σ_k V_k·(−i(V_k)) / VDD over the VDD rail and the "
           "input sources (same definition as Part B), 64 vectors × 200 Part B corners.",
           "* ML: Boolean simulation gives each gate's inputs; each gate's leakage comes from the Part B gate model "
           "(four transistor ML models, power accounting). **Gate-internal node voltages are looked up from the Part B "
           "isolated-gate SPICE references** for the same gate type / input pattern / corner — this lookup is "
           "SPICE-derived. **No full-circuit SPICE quantity is used:** the ML function only takes (vector, corner). "
           f"Prediction ran with subprocess blocked ({calls['n']} attempts) in {t_ml:.2f} s.",
           "* Errors: e = (I_ML − I_SPICE)/I_SPICE; bias = median(e); median / p95 of |e|; "
           "mean accuracy = 100 %·(1 − mean|e|); R² on log10(I) and on I.", "",
           "## Results (64 vectors × 200 corners = 12,800)", "", md(show), "",
           "![parity](fig_cla3_parity.png)", "", "![per vector](fig_cla3_error_by_vector.png)", "",
           f"Compared with isolated-gate validation (Part B, 3,600 cases): median {gate_overall['median_err']:.2%}, "
           f"p95 {gate_overall['p95_err']:.2%}, mean accuracy {gate_overall['mean_accuracy_pct']:.2f} %.", "",
           "Best vectors: " + "; ".join(f"A={int(r.A)}, B={int(r.B)} ({r.median_err:.2%})" for r in best_v.itertuples())
           + ". Worst vectors: " + "; ".join(f"A={int(r.A)}, B={int(r.B)} ({r.median_err:.2%})" for r in worst_v.itertuples())
           + ". Spearman correlation of the signed error with " + ", ".join(f"{k} {v:+.2f}" for k, v in corr.items()) + ".", "",
           "## Why circuit-level accuracy differs from isolated-gate accuracy", "",
           "1. **Composition (no ML involved):** the hierarchical sum assumes every gate input sits exactly at 0/VDD from an "
           "ideal source and ignores loading. In the circuit, gate inputs are driven by other gates' outputs, which sit "
           "slightly off the rails (up to tens of µV here) and supply the fan-out's gate-tunnelling current through their "
           "ON transistors. The diagnostic row 'sum of Part B SPICE gate leakages vs full-circuit SPICE' isolates this "
           "effect.",
           "2. **ML error per gate:** each of the 36 gate estimates carries the Part B error. Random errors partly average out "
           "in the sum (36 terms), but any common bias (e.g. from the same transistor model case at the same corner) adds "
           "coherently, because all gates share one process draw and temperature per corner.",
           "3. The two effects combine; the third table row shows the ML-only part relative to the SPICE gate sum.", "",
           "**Measured here:** the composition error is negligible (median "
           f"{comp_s['median_err']:.4%}, p95 {comp_s['p95_err']:.3%}): with leakage-level currents the internal nets "
           "sit within 0.03 % of VDD of the rails, so loading barely changes any gate's operating point. Practically all "
           f"circuit error is the transistor-ML error ({gml_s['median_err']:.2%} median vs the SPICE gate sum). The "
           f"circuit-level error ({main_s['median_err']:.2%} median, {main_s['p95_err']:.1%} p95) is *smaller* than "
           f"the isolated-gate error ({gate_overall['median_err']:.2%}, {gate_overall['p95_err']:.1%}) because the 36 "
           "per-gate errors partly cancel in the sum; what survives is their common component — the bias "
           f"({main_s['bias']:+.2%}) and its temperature dependence (Spearman ρ(e, T) = {corr['temp']:+.2f}: the ML "
           "sum is low at cold corners and closer/high at hot ones). So the PDF's expected accuracy loss from "
           "loading did not materialise for this circuit at this sizing; correlated per-gate bias is the limiting "
           "factor.", "",
           "Data: `data/circuit/cla3_spice_results.csv` (SPICE), `data/circuit/cla3_ml_predictions.csv` (ML, per vector × "
           "corner), `results/c/cla3_per_vector.csv`, `results/c/cla3_gate_case_usage.csv`.", ""]
    (OUT / "cla3_report.md").write_text("\n".join(rep), encoding="utf-8")
    (CFG.reports / "c_cla3_report.md").write_text("\n".join(rep).replace("](fig_", "](../results/c/fig_"), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(ev.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print("spearman:", {k: round(v, 3) for k, v in corr.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
