"""Detect spurious DC operating points in the SPICE references (SPICE-only check; the ML
models are NOT used to decide anything here).

Finding that motivated this: with rbodymod = 1 / rgatemod = 1 the BSIM4 devices have internal
body/gate nodes, and for some gate netlists ngspice's default operating-point path (direct
Newton) converges to a non-physical state of those internal nodes (large circulating body
current, e.g. AND2 '00' at new corner 83: 1.07e-8 A instead of 5.28e-10 A). Starting the
solve with gmin stepping (`.options noopiter`) or from logic-level node voltages (.nodeset)
gives the physical state, in which every transistor carries exactly the current of the same
isolated transistor at the same terminal voltages.

Procedure per reference record
  1. re-solve with `.options noopiter` (gmin stepping from the start);
  2. if |I_default - I_noopiter| / I_noopiter > 1e-3, the record is flagged (non-unique OP);
  3. for gate records (device probes available) both solutions are checked for DEVICE
     CONSISTENCY: each transistor's SPICE currents are compared with an isolated single-
     transistor SPICE run at identical terminal voltages / parameters. The consistent solution
     is the corrected reference.

Outputs (never overwrite the originals):
  <set>/op_check.csv            per record: default and noopiter leakage, flag
  <set>/gate_spice_results_corrected.csv | cla3_spice_results_corrected.csv
  results/improve/op_uniqueness_summary.csv

Run:  .venv/bin/python scripts/d5_op_uniqueness.py --gates data/gates data/final_eval/gates
      .venv/bin/python scripts/d5_op_uniqueness.py --cla data/circuit:data/gates data/final_eval/circuit:data/final_eval/gates
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import leakage.circuit as C  # noqa: E402
import leakage.gates as G  # noqa: E402
from leakage.config import load_config  # noqa: E402
from leakage.simulator import NgspiceSimulator  # noqa: E402
from leakage.transistor import simulate_transistors  # noqa: E402

CFG = load_config()
ALT = ".options noopiter"
TOL = 1e-3


def with_alt(fn):
    def wrapped(*a, **k):
        return fn(*a, **k).replace(".end\n", ALT + "\n.end\n")
    return wrapped


def device_consistency(dev: pd.DataFrame, corners: pd.DataFrame, sim) -> pd.Series:
    """max relative deviation of each record's transistor currents from isolated-transistor SPICE."""
    rows = []
    for r in dev.itertuples(index=False):
        c = corners.loc[r.corner_id]
        pre = "n_" if r.type == "nmos" else "p_"
        state = ("on" if (r.vg > c.vdd / 2) == (r.type == "nmos") else "off")
        rows.append({"case": f"{r.type}_{state}", "vdd": c.vdd, "temp": c.temp, "w": r.w, "vd": r.vd, "vs": r.vs,
                     "vg": c.vdd if r.vg > c.vdd / 2 else 0.0, "vb": c.vdd if r.type == "pmos" else 0.0,
                     **{p: c[pre + p] for p in ("vth0", "toxe", "ndep", "u0", "xj")}})
    iso = pd.DataFrame(rows)
    out = np.zeros(len(dev))
    for case, g in iso.groupby("case"):
        res = simulate_transistors(CFG, case, g.drop(columns="case").reset_index(drop=True), sim)
        for col in ("id", "ig", "is", "ib"):
            a = dev[col].to_numpy()[g.index]
            b = res[col].to_numpy()
            out[g.index] = np.maximum(out[g.index], np.abs(a - b) / np.maximum(np.abs(b), 1e-14))
    return pd.Series(out, index=dev.index)


def check_gates(d: Path, sim) -> dict:
    ref = pd.read_csv(d / "gate_spice_results.csv", dtype={"inputs": str})
    devs = pd.read_csv(d / "gate_spice_devices.csv", dtype={"inputs": str})
    corners = pd.read_csv(d / "corners.csv").set_index("corner_id")
    gates = G.build_gates(CFG)
    orig = G.gate_circuit
    G.gate_circuit = with_alt(orig)
    try:
        alt_s, alt_d = [], []
        for name, g in gates.items():
            s, dd, _ = G.simulate_gate(CFG, g, corners.reset_index(), sim)
            alt_s.append(s); alt_d.append(dd)
    finally:
        G.gate_circuit = orig
    alt = pd.concat(alt_s, ignore_index=True)
    altd = pd.concat(alt_d, ignore_index=True)
    key = ["gate", "inputs", "corner_id"]
    m = ref[key + ["i_leak"]].merge(alt[key + ["i_leak", "status", "logic_ok"]].rename(
        columns={"i_leak": "i_leak_alt", "status": "status_alt", "logic_ok": "logic_alt"}), on=key)
    m["rel_diff"] = (m.i_leak - m.i_leak_alt).abs() / m.i_leak_alt.abs()
    m["flag"] = m.rel_diff > TOL
    # device consistency for flagged records, both solutions
    m["dev_dev_default"] = np.nan
    m["dev_dev_alt"] = np.nan
    fl = m[m.flag]
    if len(fl):
        for label, src in (("default", devs), ("alt", altd)):
            sub = src.merge(fl[key], on=key).reset_index(drop=True)
            dv = device_consistency(sub, corners, sim)
            worst = sub.assign(dev=dv).groupby(key).dev.max().reset_index()
            m = m.merge(worst.rename(columns={"dev": f"_w_{label}"}), on=key, how="left")
            m[f"dev_dev_{label}"] = m[f"_w_{label}"]
            m = m.drop(columns=f"_w_{label}")
    m["corrected_source"] = np.where(~m.flag, "default",
                                     np.where(m.dev_dev_alt < m.dev_dev_default, "noopiter", "default"))
    m.to_csv(d / "op_check.csv", index=False)
    corr = ref.copy()
    use = m.set_index(key).corrected_source == "noopiter"
    idx = corr.set_index(key).index
    pick = use.reindex(idx).fillna(False).to_numpy()
    altv = alt.set_index(key).reindex(idx)
    for col in [c for c in ref.columns if c in alt.columns and c not in key]:
        corr[col] = np.where(pick, altv[col].to_numpy(dtype=object), corr[col].to_numpy(dtype=object))
        try:
            corr[col] = pd.to_numeric(corr[col])
        except (ValueError, TypeError):
            pass
    corr["op_corrected"] = pick
    corr.to_csv(d / "gate_spice_results_corrected.csv", index=False)
    return {"set": str(d), "kind": "gates", "records": len(m), "flagged": int(m.flag.sum()),
            "corrected": int(pick.sum()), "max_rel_diff": float(m.rel_diff.max()),
            "flagged_list": "; ".join(f"{r.gate} {r.inputs} c{r.corner_id} ({r.i_leak:.3e} vs {r.i_leak_alt:.3e}; "
                                      f"device dev default {r.dev_dev_default:.2g} / alt {r.dev_dev_alt:.2g})"
                                      for r in m[m.flag].itertuples())}


def check_cla(d: Path, gate_dir: Path, sim, sfx: str = "") -> dict:
    ref = pd.read_csv(d / f"cla3_spice_results{sfx}.csv")
    corners = pd.read_csv(gate_dir / "corners.csv")
    circ = C.build_cla3()
    orig = C.circuit_netlist
    jobs = [(int(r.vector_id), corners.set_index("corner_id").loc[int(r.corner_id)].to_dict() | {"corner_id": int(r.corner_id)})
            for r in ref.itertuples()]
    circuits = [with_alt(orig)(CFG, circ, C.vector_bits(v), c) for v, c in jobs]
    res = sim.run_op(circuits, C.circuit_vectors(circ))
    alt = []
    for k, (v, c) in enumerate(jobs):
        ok = res.status.iloc[k] == "ok"
        alt.append(C.leakage_from_sources(circ, C.vector_bits(v), float(c["vdd"]), res.values.iloc[k])["i_leak"]
                   if ok else np.nan)
    m = ref[["vector_id", "corner_id", "i_leak"]].copy()
    m["i_leak_alt"] = alt
    m["rel_diff"] = (m.i_leak - m.i_leak_alt).abs() / m.i_leak_alt.abs()
    m["flag"] = m.rel_diff > TOL
    m.to_csv(d / f"op_check{sfx}.csv", index=False)
    return {"set": str(d) + sfx, "kind": "CLA3", "records": len(m), "flagged": int(m.flag.sum()),
            "max_rel_diff": float(m.rel_diff.max()),
            "flagged_list": "; ".join(f"v{int(r.vector_id)} c{int(r.corner_id)} ({r.i_leak:.3e} vs {r.i_leak_alt:.3e})"
                                      for r in m[m.flag].head(30).itertuples())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates", nargs="*", default=[])
    ap.add_argument("--cla", nargs="*", default=[], help="circuit_dir:gate_dir[:suffix] (suffix _cin1 = Cin = 1 half)")
    ap.add_argument("--workers", type=int, default=CFG.ngspice_workers)
    a = ap.parse_args()
    sim = NgspiceSimulator.from_config(CFG, workers=a.workers, chunk=40)
    rows = []
    for d in a.gates:
        t0 = time.perf_counter()
        r = check_gates(Path(d), sim)
        print(f"{d}: {r['flagged']}/{r['records']} flagged, corrected {r['corrected']}, max rel diff "
              f"{r['max_rel_diff']:.2e} ({time.perf_counter() - t0:.0f}s)\n   {r['flagged_list']}", flush=True)
        rows.append(r)
    for pair in a.cla:
        cd, gd, *sx = pair.split(":")
        t0 = time.perf_counter()
        r = check_cla(Path(cd), Path(gd), sim, sx[0] if sx else "")
        print(f"{cd}: {r['flagged']}/{r['records']} flagged, max rel diff {r['max_rel_diff']:.2e} "
              f"({time.perf_counter() - t0:.0f}s)\n   {r['flagged_list']}", flush=True)
        rows.append(r)
    out = CFG.results / "improve" / "op_uniqueness_summary.csv"
    old = pd.read_csv(out) if out.exists() else pd.DataFrame()
    pd.concat([old, pd.DataFrame(rows)], ignore_index=True).drop_duplicates("set", keep="last").to_csv(out, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
