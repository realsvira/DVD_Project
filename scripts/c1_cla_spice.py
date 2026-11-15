"""Part C, SPICE reference: the complete transistor-level 3-bit CLA (36 gates, 192 transistors)
at the same 200 PVT corners as Part B. All 128 input combinations (A, B, Cin) are simulated
in two halves: --cin 0 (vector_id 0..63, the original run) and --cin 1 (vector_id 64..127).

  --smoke   nominal corner + corner 0, the 64 vectors of the half (128 sims); checks EVERY net
  (default) smoke first, then 64 x 200 = 12,800 full-circuit simulations
  --cin 1   the Cin = 1 half; written to *_cin1 files, the Cin = 0 files are never touched

Outputs (suffix _cin1 for --cin 1)
  data/circuit/cla3_spice_results.csv   leakage, source currents, output voltages, checks
  data/circuit/cla3_failures.csv        failed / logic-error simulations (empty if none)
  data/circuit/cla3_gate_list.csv       every gate instance, its inputs, output, transistors (--cin 0)
  data/circuit/metadata.json            seeds, hashes, counts, warnings
  netlists/circuit/CLA3.cir             netlist (A=5, B=3, Cin=0, corner 0) with .control (--cin 0)
  results/c/cla3_spice_summary.md

Run:  .\\.venv\\Scripts\\python.exe scripts\\c1_cla_spice.py --smoke
      .\\.venv\\Scripts\\python.exe scripts\\c1_cla_spice.py
      .\\.venv\\Scripts\\python.exe scripts\\c1_cla_spice.py --cin 1
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from leakage.circuit import (adder_result, build_cla3, circuit_netlist, circuit_vectors, device_list,  # noqa: E402
                             evaluate, leakage_from_sources, vector_bits)
from leakage.config import load_config  # noqa: E402
from leakage.gates import CORNER_COLUMNS, build_gates  # noqa: E402
from leakage.simulator import NgspiceSimulator  # noqa: E402
from leakage.spice import ngspice_version  # noqa: E402
from leakage.transistor import nominal_process  # noqa: E402

CFG = load_config()
OUT = CFG.data / "circuit"
RES = CFG.results / "c"
TOL = float(CFG.raw["gates"].get("logic_tolerance", 0.05))


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def nominal_corner() -> pd.Series:
    r = {"corner_id": -1, "vdd": CFG.vdd_nominal, "temp": 27.0}
    for d, pre in (("nmos", "n_"), ("pmos", "p_")):
        for p, v in nominal_process(CFG, d).items():
            r[pre + p] = v
    return pd.Series(r)[CORNER_COLUMNS]


def simulate(circ, corners: pd.DataFrame, sim, label, vids=range(64)):
    jobs = [(vid, c) for _, c in corners.iterrows() for vid in vids]
    t0 = time.perf_counter()
    res = sim.run_op([circuit_netlist(CFG, circ, vector_bits(vid), c) for vid, c in jobs], circuit_vectors(circ),
                     progress=lambda d, n, e: print(f"  [{label}] chunk {d}/{n}  {e:.0f}s", flush=True)
                     if d % max(1, n // 10) == 0 or d == n else None)
    rows = []
    for k, (vid, c) in enumerate(jobs):
        bits = vector_bits(vid)
        exp = evaluate(circ, bits)
        r = res.values.iloc[k]
        vdd = float(c.vdd)
        ok = res.status.iloc[k] == "ok"
        lk = leakage_from_sources(circ, bits, vdd, r) if ok else {"p_leak": np.nan, "i_leak": np.nan,
                                                                  "i_vdd": np.nan, "i_inputs_total": np.nan}
        margins = [abs(float(r[f"v({g.output})"]) - exp[g.output] * vdd) / vdd for g in circ.gates] if ok else [np.nan]
        sum_spice = (sum((float(r[f"v({n})"]) > vdd / 2) << i for i, n in enumerate(("s0", "s1", "s2", "c3")))
                     if ok else -1)
        rows.append({"vector_id": vid, "A": vid % 8, "B": (vid // 8) % 8, "cin": bits["cin"],
                     "corner_id": int(c.corner_id),
                     **{col: c[col] for col in CORNER_COLUMNS if col != "corner_id"}, **lk,
                     **{f"v_{n}": float(r[f"v({n})"]) for n in circ.outputs},
                     "expected_sum": adder_result(bits), "spice_sum": sum_spice,
                     "max_node_deviation_vdd": float(np.max(margins)),
                     "all_nodes_logic_ok": bool(ok and np.max(margins) <= TOL),
                     "sum_ok": sum_spice == adder_result(bits),
                     "status": res.status.iloc[k], "messages": " | ".join(res.messages[k])})
    print(f"  [{label}] {len(jobs)} sims in {time.perf_counter() - t0:.0f}s")
    return pd.DataFrame(rows), res


def checks(s, n):
    return [(len(s) == n, f"{len(s)} simulations == {n}"),
            ((s.status == "ok").all(), f"converged {int((s.status == 'ok').sum())}/{len(s)}"),
            (s.sum_ok.all(), f"A+B correct at the outputs {int(s.sum_ok.sum())}/{len(s)}"),
            (s.all_nodes_logic_ok.all(), f"all 36 internal nets within {TOL:.0%} VDD of their logic level "
                                         f"{int(s.all_nodes_logic_ok.sum())}/{len(s)} (max dev {s.max_node_deviation_vdd.max():.2e} VDD)"),
            (bool((s.i_leak > 0).all()), "leakage positive and finite")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=CFG.ngspice_workers)
    ap.add_argument("--corners-file", type=Path, default=CFG.data / "gates" / "corners.csv")
    ap.add_argument("--out-data", type=Path, default=None, help="default data/circuit")
    ap.add_argument("--out-results", type=Path, default=None, help="default results/c")
    ap.add_argument("--no-netlists", action="store_true")
    ap.add_argument("--cin", type=int, choices=(0, 1), default=0, help="which half of the 128 vectors")
    a = ap.parse_args()
    sfx = "" if a.cin == 0 else "_cin1"
    vids = range(64 * a.cin, 64 * a.cin + 64)
    circ = build_cla3()
    sim = NgspiceSimulator.from_config(CFG, workers=a.workers, chunk=32)
    ver = ngspice_version(sim.exe)
    global OUT, RES
    OUT = (a.out_data or OUT).resolve()
    RES = (a.out_results or RES).resolve()
    corners = pd.read_csv(a.corners_file)
    print(f"CLA3: {len(circ.gates)} gates, {len(device_list(CFG, circ))} transistors; ngspice {ver}; "
          f"corners from data/gates/corners.csv ({len(corners)})")

    smoke_c = pd.DataFrame([nominal_corner(), corners.iloc[0]])
    s, _ = simulate(circ, smoke_c, sim, "smoke", vids)
    ck = checks(s, 128)
    for p, t in ck:
        print(f"  [{'PASS' if p else 'FAIL'}] {t}")
    RES.mkdir(parents=True, exist_ok=True)
    s.to_csv(RES / f"cla3_smoke{sfx}.csv", index=False)
    if not all(p for p, _ in ck):
        print(s[~s.sum_ok | ~s.all_nodes_logic_ok][["vector_id", "corner_id", "expected_sum", "spice_sum",
                                                      "max_node_deviation_vdd", "messages"]].head(20).to_string())
        print("SMOKE TEST FAILED")
        return 1
    nom = s[s.corner_id == -1]
    print(f"SMOKE TEST PASSED (nominal corner leakage {nom.i_leak.min():.3e} .. {nom.i_leak.max():.3e} A)\n")
    if a.smoke:
        return 0

    # ---------------------------------------------------------------- documentation files
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cin == 0:
        write_docs(circ, corners, a.no_netlists)
    return full_run(circ, corners, sim, ver, vids, sfx, a)


def write_docs(circ, corners, no_netlists):
    gates = build_gates(CFG)
    pd.DataFrame([{"instance": g.name, "type": g.kind, "input_a": g.inputs[0],
                   "input_b": g.inputs[1] if len(g.inputs) > 1 else "", "output": g.output,
                   "transistors": len(gates[g.kind].devices)} for g in circ.gates]).to_csv(
        OUT / "cla3_gate_list.csv", index=False)
    nl = (CFG.results / "scratch_netlists") if no_netlists else CFG.netlists / "circuit"
    nl.mkdir(parents=True, exist_ok=True)
    ex = circuit_netlist(CFG, circ, vector_bits(5 + 8 * 3), corners.iloc[0],
                         title="* 3-bit carry-lookahead adder, complete transistor netlist (36 gates, 192 MOSFETs)\n"
                               "* Vector A=5 (a2a1a0=101), B=3 (b2b1b0=011), Cin=0; corner 0. Expected S=8 (c3s2s1s0=1000)\n"
                               "* Leakage: I_leak = sum_k V_k*(-i(V_k))/VDD over Vdd and the input sources")
    ex = ex.rsplit(".end", 1)[0] + (".control\nset numdgt=10\nop\nprint v(s0) v(s1) v(s2) v(c3) i(vdd)\n"
                                    ".endc\n.end\n")
    (nl / "CLA3.cir").write_text(ex, encoding="utf-8")


def full_run(circ, corners, sim, ver, vids, sfx, a):
    print(f"== full: 64 vectors (Cin = {a.cin}) x {len(corners)} corners = {64 * len(corners)} "
          "full-circuit simulations ==")
    t0 = time.perf_counter()
    s, res = simulate(circ, corners, sim, "full", vids)
    wall = time.perf_counter() - t0
    s.to_csv(OUT / f"cla3_spice_results{sfx}.csv", index=False)
    s[(s.status != "ok") | ~s.sum_ok | ~s.all_nodes_logic_ok].to_csv(OUT / f"cla3_failures{sfx}.csv", index=False)
    ck = checks(s, 64 * len(corners))
    meta = {"created": dt.datetime.now().isoformat(timespec="seconds"), "ngspice": ver,
            "circuit": "3-bit carry-lookahead adder (CLA3)", "gates": len(circ.gates),
            "transistors": len(device_list(CFG, circ)), "cin": f"logic {a.cin} (source at {'VDD' if a.cin else '0 V'})",
            "vectors": f"vector_id = A + 8*B + 64*Cin, A,B in 0..7, Cin = {a.cin}", "corners_sha256": sha256(a.corners_file),
            "model_sha256": sha256(CFG.spice_model_file), "options": CFG.spice_options,
            "simulations": len(s), "failed": int((s.status != "ok").sum()), "wall_seconds": round(wall, 1),
            "simulator_warnings": dict(res.batch_warnings),
            "checks": [{"pass": bool(p), "check": t} for p, t in ck],
            "results_sha256": sha256(OUT / f"cla3_spice_results{sfx}.csv")}
    (OUT / f"metadata{sfx}.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    lines = [f"# Part C: CLA3 full-circuit SPICE reference (Cin = {a.cin})", "",
             f"{len(s)} simulations (64 vectors × {len(corners)} corners), ngspice {ver}, {wall:.0f} s.", ""] + \
            [f"- {'PASS' if p else 'FAIL'}: {t}" for p, t in ck] + \
            ["", f"Leakage over all simulations: median {s.i_leak.median():.3e} A, "
                 f"min {s.i_leak.min():.3e} A, max {s.i_leak.max():.3e} A.", ""]
    (RES / f"cla3_spice_summary{sfx}.md").write_text("\n".join(lines), encoding="utf-8")
    for p, t in ck:
        print(f"  [{'PASS' if p else 'FAIL'}] {t}")
    return 0 if all(p for p, _ in ck) else 1


if __name__ == "__main__":
    raise SystemExit(main())
