"""Part B, step 3: transistor-level SPICE reference simulation of INV, NAND2, NOR2,
AND2 and OR2 for every input combination at 200 shared PVT corners (18 x 200 = 3,600).

These results are evaluation references only. NO model is trained on them.

Modes
  --smoke   nominal corner only (VDD = 1.0 V, 27 C, nominal process) for all 18 cases,
            plus corner 0: checks truth tables, power balance, convergence. Run this first.
  (default) full run over all corners; the smoke test runs first and must pass.

Outputs
  data/gates/corners.csv                 200 PVT corners (corner_id shared by Part B and C)
  data/gates/gate_spice_results.csv      3,600 rows: leakage, node voltages, checks
  data/gates/gate_spice_devices.csv      per simulation x transistor: terminal V and SPICE I
  data/gates/failures.csv                simulator failures / logic errors (empty if none)
  data/gates/metadata.json               seeds, sizing, versions, hashes, counts
  netlists/gates/<GATE>.cir              one example netlist per gate (corner 0)
  results/b3/truth_table.csv / summary_by_case.csv / b3_report.md

Run:  .venv/bin/python scripts/b3_gate_spice.py --smoke
      .venv/bin/python scripts/b3_gate_spice.py
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
from leakage.config import load_config  # noqa: E402
from leakage.gates import (CORNER_COLUMNS, build_gates, example_netlist, sample_corners,  # noqa: E402
                           simulate_gate)
from leakage.simulator import NgspiceSimulator  # noqa: E402
from leakage.spice import ngspice_version  # noqa: E402
from leakage.transistor import nominal_process  # noqa: E402

CFG = load_config()
OUT_DATA = CFG.data / "gates"
OUT_RES = CFG.results / "b3"
POWER_TOL = 1e-6


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def nominal_corner() -> pd.DataFrame:
    r = {"corner_id": -1, "vdd": CFG.vdd_nominal, "temp": 27.0}
    for d, pre in (("nmos", "n_"), ("pmos", "p_")):
        for p, v in nominal_process(CFG, d).items():
            r[pre + p] = v
    return pd.DataFrame([r])[CORNER_COLUMNS]


def run(gates, corners, sim, label):
    summ, devs, n_warn = [], [], {}
    t0 = time.perf_counter()
    for name, g in gates.items():
        s, d, res = simulate_gate(CFG, g, corners, sim)
        summ.append(s); devs.append(d)
        n_warn[name] = dict(res.batch_warnings)
        print(f"  [{label}] {name:6s} {len(s):5d} sims  failed={int((s.status != 'ok').sum())}  "
              f"logic_ok={int(s.logic_ok.sum())}/{len(s)}  max power-balance err={s.power_balance_rel.max():.1e}  "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)
    return pd.concat(summ, ignore_index=True), pd.concat(devs, ignore_index=True), n_warn


def checks(s: pd.DataFrame, n_expected: int) -> list[tuple[bool, str]]:
    return [
        (len(s) == n_expected, f"{len(s)} simulations == {n_expected} expected"),
        ((s.status == "ok").all(), f"converged: {int((s.status == 'ok').sum())}/{len(s)}"),
        (s.logic_ok.all(), f"truth table correct: {int(s.logic_ok.sum())}/{len(s)}"),
        ((s.power_balance_rel < POWER_TOL).all(),
         f"power balance sources == devices (max rel err {s.power_balance_rel.max():.1e} < {POWER_TOL:g})"),
        ((s.i_leak > 0).all() and np.isfinite(s.i_leak).all(), "leakage positive and finite"),
        (not s.duplicated(["gate", "inputs", "corner_id"]).any(), "unique (gate, inputs, corner)"),
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--corner-seed", type=int, default=None,
                    help="different seed = independent corner set (default: config seeds.corners)")
    ap.add_argument("--out-data", type=Path, default=None, help="default data/gates")
    ap.add_argument("--out-results", type=Path, default=None, help="default results/b3")
    ap.add_argument("--no-netlists", action="store_true", help="do not (re)write netlists/gates/*.cir")
    a = ap.parse_args()
    gates = build_gates(CFG)
    sim = NgspiceSimulator.from_config(CFG, workers=a.workers, chunk=50)
    ver = ngspice_version(sim.exe)
    print(f"ngspice {ver}; gates: " + ", ".join(
        f"{g.name}[{', '.join(f'{d.name}:{d.dev[0].upper()}{d.w*1e9:.0f}n' for d in g.devices)}]"
        for g in gates.values()))

    # ---------------------------------------------------------------- smoke
    global OUT_DATA, OUT_RES
    OUT_DATA = (a.out_data or OUT_DATA).resolve()
    OUT_RES = (a.out_results or OUT_RES).resolve()
    corners = sample_corners(CFG, seed=a.corner_seed)
    smoke_c = pd.concat([nominal_corner(), corners.head(1)], ignore_index=True)
    print("== smoke: nominal corner + corner 0, all 18 gate/input cases ==")
    s, d, _ = run(gates, smoke_c, sim, "smoke")
    ok = checks(s, 18 * 2)
    for p, t in ok:
        print(f"  [{'PASS' if p else 'FAIL'}] {t}")
    tt = s[s.corner_id == -1][["gate", "inputs", "v_out", *[c for c in s.columns if c.startswith("v_") and c != "v_out"],
                               "logic_ok", "i_leak", "i_vdd"]]
    pd.set_option("display.width", 220)
    print("\nNominal corner truth table (V) and leakage (A):")
    print(tt.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    OUT_RES.mkdir(parents=True, exist_ok=True)
    s.to_csv(OUT_RES / "smoke_results.csv", index=False)
    if not all(p for p, _ in ok):
        print("SMOKE TEST FAILED: fix the gates before the full run")
        print(s[(s.status != "ok") | ~s.logic_ok][["gate", "inputs", "corner_id", "messages"]].to_string())
        return 1
    print("SMOKE TEST PASSED\n")
    if a.smoke:
        return 0

    # ---------------------------------------------------------------- full
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    corners.to_csv(OUT_DATA / "corners.csv", index=False)
    nl = CFG.netlists / "gates"
    nl.mkdir(parents=True, exist_ok=True)
    for g in ([] if a.no_netlists else gates.values()):
        (nl / f"{g.name}.cir").write_text(example_netlist(CFG, g, g.patterns[0], corners.iloc[0]), encoding="utf-8")

    n_cases = sum(len(g.patterns) for g in gates.values())
    print(f"== full: {n_cases} gate/input cases x {len(corners)} corners = {n_cases * len(corners)} simulations ==")
    t0 = time.perf_counter()
    s, d, warns = run(gates, corners, sim, "full")
    wall = time.perf_counter() - t0
    s.to_csv(OUT_DATA / "gate_spice_results.csv", index=False)
    d.to_csv(OUT_DATA / "gate_spice_devices.csv", index=False)
    bad = s[(s.status != "ok") | ~s.logic_ok | (s.power_balance_rel >= POWER_TOL)]
    bad.to_csv(OUT_DATA / "failures.csv", index=False)
    ck = checks(s, n_cases * len(corners))

    truth = (s.groupby(["gate", "inputs"])
             .agg(n=("logic_ok", "size"), logic_ok=("logic_ok", "sum"),
                  v_out_min=("v_out", "min"), v_out_max=("v_out", "max")).reset_index())
    q = lambda x: x.quantile(0.05)  # noqa: E731
    q.__name__ = "p5"
    summ = (s.groupby(["gate", "inputs"]).i_leak
            .agg(["median", "mean", "min", q, lambda x: x.quantile(0.95), "max"])
            .rename(columns={"<lambda_0>": "p95", "p5": "p5"}).reset_index())
    truth.to_csv(OUT_RES / "truth_table.csv", index=False)
    summ.to_csv(OUT_RES / "summary_by_case.csv", index=False)

    meta = {"created": dt.datetime.now().isoformat(timespec="seconds"), "ngspice": ver,
            "model_sha256": sha256(CFG.spice_model_file), "options": CFG.spice_options,
            "corners": {"n": len(corners), "seed": CFG.seeds["corners"],
                        "rng": "default_rng([seed, corner_id]); VDD, T, NMOS x5, PMOS x5"},
            "sizing": {g.name: {d.name: {"type": d.dev, "w_nm": round(d.w * 1e9, 3), "d": d.d, "g": d.g,
                                         "s": d.s, "b": d.b} for d in g.devices} for g in gates.values()},
            "leakage_definition": "I_leak = sum_k V_k*(-i(V_k)) / VDD over Vdd and input sources",
            "simulations": len(s), "failed": int((s.status != "ok").sum()),
            "logic_errors": int((~s.logic_ok).sum()), "wall_seconds": round(wall, 1),
            "simulator_warnings": warns, "checks": [{"pass": bool(p), "check": t} for p, t in ck],
            "results_sha256": sha256(OUT_DATA / "gate_spice_results.csv"),
            "corners_sha256": sha256(OUT_DATA / "corners.csv")}
    (OUT_DATA / "metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    def md(df):
        cols = list(df.columns)
        rows = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
        for _, r in df.iterrows():
            rows.append("| " + " | ".join(f"{v:.4g}" if isinstance(v, (float, np.floating)) else str(v)
                                          for v in r) + " |")
        return "\n".join(rows)

    sizing = pd.DataFrame([{"gate": g.name, "device": d.name, "type": d.dev, "W (nm)": round(d.w * 1e9),
                            "d": d.d, "g": d.g, "s": d.s, "b": d.b} for g in gates.values() for d in g.devices])
    rep = ["# Part B step 3: gate-level SPICE references", "",
           f"Generated {meta['created']}, ngspice {ver}, model `{CFG.spice_model_file.name}`. "
           f"{len(s)} simulations ({n_cases} gate/input cases x {len(corners)} corners) in {wall:.0f} s.",
           "These results are evaluation references only; no model is trained on them.", "",
           "## Checks", ""] + [f"- {'PASS' if p else 'FAIL'}: {t}" for p, t in ck] + [
           "", "## Leakage definition", "",
           "Each source V_k (VDD rail and each input) delivers I_k = −i(V_k). "
           "P_leak = Σ V_k·I_k and **I_leak = P_leak / VDD** (positive). It includes sub-threshold, "
           "gate-tunnelling (also from inputs held at VDD) and junction leakage. The VDD-rail current "
           "alone (`i_vdd`) and the input currents (`i_in_a`, `i_in_b`) are stored too. Every device "
           "terminal has a 0 V probe; Σ over devices of V·I equals P_leak (power-balance check above).", "",
           "## Sizing and topology (W not specified by the PDF)", "",
           "Equal-drive sizing: unit inverter Wn = 2·Lmin = 90 nm, Wp = 4·Lmin = 180 nm; series stacks "
           "of two use 2x unit width; parallel devices unit width. NAND2: MNA (gate A) on top of MNB "
           "(gate B); NOR2: MPA (gate A) at VDD above MPB (gate B). AND2/OR2 = NAND2/NOR2 + inverter, "
           "simulated as one complete netlist.", "", md(sizing), "",
           "## Truth table (all corners)", "", md(truth), "",
           "## Leakage per gate and input (A, over 200 corners)", "", md(summ), ""]
    (OUT_RES / "b3_report.md").write_text("\n".join(rep), encoding="utf-8")
    for p, t in ck:
        print(f"  [{'PASS' if p else 'FAIL'}] {t}")
    print(summ.to_string(index=False, float_format=lambda x: f"{x:.3e}"))
    return 0 if all(p for p, _ in ck) else 1


if __name__ == "__main__":
    raise SystemExit(main())
