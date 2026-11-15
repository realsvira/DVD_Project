"""Part A1 smoke test: all four transistor cases at nominal conditions,
plus a corner sweep over the edges of the sampling ranges.

1. Nominal point per case: VDD = 1.0 V, T = 27 C, W = 2*Lmin = 90 nm, nominal
   process; full Vds across the device (NMOS Vd=VDD, Vs=0; PMOS Vd=0, Vs=VDD).
   * writes the stand-alone example netlist netlists/transistor/<case>.cir
   * simulates it with the batch engine AND by running the example netlist
     on its own; both must agree
   * checks: finite currents, Kirchhoff (sum of 4 currents ~ 0), sign of Id,
     ON current >> OFF current
2. Corner sweep per case: T in {-40, 125} C x VDD in {0.9, 1.1} V x
   W in {1, 8}*Lmin x process all -10 % / all +10 % x 3 (Vd, Vs) patterns
   = 48 circuits per case; checks convergence, finiteness and Kirchhoff.

Outputs: results/a1_smoke_test.csv, results/a1_corner_sweep.csv,
         reports/a1_smoke_test.md, netlists/transistor/*.cir
Run:     .venv/bin/python scripts/a1_smoke_test.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from leakage.config import load_config  # noqa: E402
from leakage.simulator import NgspiceSimulator  # noqa: E402
from leakage.spice import ngspice_version  # noqa: E402
from leakage.transistor import (CURRENT_COLS, KCL_ABS, KCL_REL, example_netlist,  # noqa: E402
                                kcl_ok, kcl_residual, make_inputs, nominal_process,
                                simulate_standalone, simulate_transistors)


def main() -> int:
    cfg = load_config()
    sim = NgspiceSimulator.from_config(cfg)
    checks: list[tuple[bool, str]] = []
    print(f"ngspice {ngspice_version(sim.exe)} | model {cfg.spice_model_file.name} | "
          f"L = {cfg.lmin*1e9:.0f} nm | workers {sim.workers}\n")

    # ---------------------------------------------------------------- 1. nominal
    out_dir = cfg.netlists / "transistor"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in cfg.cases:
        r = make_inputs(cfg, case)
        (out_dir / f"{case}.cir").write_text(example_netlist(cfg, case, r))
        eng = simulate_transistors(cfg, case, pd.DataFrame([r]), sim).iloc[0]
        ref = simulate_standalone(cfg, sim.exe, case, r)
        rows.append({"case": case, **{k: r[k] for k in ("vdd", "temp", "w", "vd", "vs", "vg", "vb")},
                     **{c: eng[c] for c in CURRENT_COLS}, "status": eng.status,
                     "standalone_max_rel_diff": float(np.max(np.abs(eng[CURRENT_COLS].to_numpy(float) - ref)
                                                              / np.abs(ref)))})
    nom = pd.DataFrame(rows)
    nom["kcl_sum"] = kcl_residual(nom)
    nom["kcl_rel"] = nom.kcl_sum.abs() / nom[CURRENT_COLS].abs().max(axis=1)

    pd.set_option("display.width", 200)
    print("Nominal operating points (A, positive = INTO the device terminal):")
    print(nom[["case", "vd", "vs", "vg", "vb", *CURRENT_COLS, "kcl_sum", "kcl_rel",
               "standalone_max_rel_diff"]].to_string(index=False, float_format=lambda x: f"{x: .4e}"))
    print()

    checks.append(((nom.status == "ok").all(), "all 4 nominal simulations converged"))
    checks.append((np.isfinite(nom[CURRENT_COLS].to_numpy(float)).all(), "all currents finite"))
    checks.append((kcl_ok(nom).all(), f"Kirchhoff: |Id+Ig+Is+Ib| <= {KCL_REL:g}*max|I| + {KCL_ABS:g} A"))
    checks.append(((nom.standalone_max_rel_diff < 1e-9).all(),
                   "batch engine == stand-alone example netlist (rel diff < 1e-9)"))
    nmos = nom.case.str.startswith("nmos")
    checks.append(((nom.loc[nmos, "id"] > 0).all() and (nom.loc[~nmos, "id"] < 0).all(),
                   "sign of Id: NMOS current enters drain, PMOS current leaves drain"))
    for dev in ("nmos", "pmos"):
        on = abs(nom.set_index("case").loc[f"{dev}_on", "id"])
        off = abs(nom.set_index("case").loc[f"{dev}_off", "id"])
        checks.append((on > 1e3 * off, f"{dev.upper()} Ion/Ioff = {on/off:.3g} (> 1e3)"))

    # ---------------------------------------------------------------- 2. corners
    corner_rows = []
    for case in cfg.cases:
        dev = cfg.cases[case].device
        nomp = nominal_process(cfg, dev)
        for t, vdd, wf, pf, pat in itertools.product(
                cfg.temp_range, cfg.vdd_range, (1, 8), (0.9, 1.1),
                ("d_high", "s_high", "mid")):
            vd, vs = {"d_high": (vdd, 0.0), "s_high": (0.0, vdd), "mid": (vdd / 2, vdd / 2)}[pat]
            r = make_inputs(cfg, case, vdd=vdd, temp=t, w=wf * cfg.lmin, vd=vd, vs=vs,
                            process={p: v * pf for p, v in nomp.items()})
            corner_rows.append({"case": case, "proc_factor": pf, "pattern": pat, **r})
    cdf = pd.DataFrame(corner_rows)
    parts = []
    for case, g in cdf.groupby("case", sort=False):
        res = simulate_transistors(cfg, case, g.drop(columns=["case", "proc_factor", "pattern"]), sim)
        res.insert(0, "case", case)
        res.insert(1, "proc_factor", g.proc_factor.to_numpy())
        res.insert(2, "pattern", g.pattern.to_numpy())
        parts.append(res)
    corners = pd.concat(parts, ignore_index=True)
    corners["kcl_sum"] = kcl_residual(corners)
    ok = corners.status == "ok"
    summary = corners.groupby("case", sort=False).apply(lambda g: pd.Series({
        "n": len(g), "failed": int((g.status != "ok").sum()),
        "min|Id|": g["id"].abs().min(), "max|Id|": g["id"].abs().max(),
        "max|Ig|": g["ig"].abs().max(), "max|Ib|": g["ib"].abs().max(),
        "max|KCL sum|": g.kcl_sum.abs().max()}), include_groups=False)
    print("Corner sweep (48 corners per case):")
    print(summary.to_string(float_format=lambda x: f"{x:.3e}"))
    print()
    checks.append((ok.all(), f"corner sweep: {int(ok.sum())}/{len(corners)} converged"))
    checks.append((kcl_ok(corners[ok]).all(), "corner sweep: Kirchhoff holds at every corner"))

    # ---------------------------------------------------------------- outputs
    cfg.results.mkdir(parents=True, exist_ok=True)
    nom.to_csv(cfg.results / "a1_smoke_test.csv", index=False)
    corners.to_csv(cfg.results / "a1_corner_sweep.csv", index=False)

    print("Checks:")
    for passed, text in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {text}")
    all_ok = all(p for p, _ in checks)
    print("\nSMOKE TEST " + ("PASSED" if all_ok else "FAILED"))

    md = ["# Part A1 smoke test", "",
          f"ngspice {ngspice_version(sim.exe)}, model `{cfg.spice_model_file.name}`, "
          f"L = {cfg.lmin*1e9:.0f} nm. Currents in A, positive = into the device terminal.", "",
          "## Nominal (VDD = 1.0 V, T = 27 C, W = 90 nm, nominal process)", "",
          nom[["case", "vd", "vs", "vg", "vb", *CURRENT_COLS, "kcl_sum"]]
          .to_markdown(index=False, floatfmt=".4e") if _has_tabulate() else
          "```\n" + nom.to_string(index=False) + "\n```", "",
          "## Corner sweep", "", "```", summary.to_string(float_format=lambda x: f"{x:.3e}"), "```", "",
          "## Checks", ""] + [f"- {'PASS' if p else 'FAIL'}: {t}" for p, t in checks] + \
         ["", f"**{'PASSED' if all_ok else 'FAILED'}**", ""]
    (cfg.reports / "a1_smoke_test.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote {len(cfg.cases)} example netlists to {out_dir}")
    return 0 if all_ok else 1


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
