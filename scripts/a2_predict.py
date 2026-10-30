"""Predict the four terminal currents of one transistor with the trained models,
optionally comparing against a fresh ngspice simulation.

Examples (PowerShell, project root):
  .\\.venv\\Scripts\\python.exe scripts\\a2_predict.py --device nmos --vd 1.0 --vg 0 --vs 0 --vb 0 --vdd 1.0 --temp 27 --w 90e-9
  .\\.venv\\Scripts\\python.exe scripts\\a2_predict.py --device pmos --vd 0 --vg 1.05 --vs 1.05 --vb 1.05 --vdd 1.05 --temp 85 --w 135e-9 --vth0 -0.47 --spice
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from leakage.inference import LeakagePredictor  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Predict Id, Ig, Is, Ib for one transistor")
    ap.add_argument("--device", required=True, choices=["nmos", "pmos"])
    ap.add_argument("--state", choices=["on", "off"], default=None, help="default: inferred from Vg")
    for v in ("vd", "vg", "vs", "vb", "vdd", "temp", "w"):
        ap.add_argument(f"--{v}", type=float, required=True)
    for p in ("vth0", "toxe", "ndep", "u0", "xj"):
        ap.add_argument(f"--{p}", type=float, default=None, help="default: nominal")
    ap.add_argument("--spice", action="store_true", help="also simulate with ngspice and compare")
    a = ap.parse_args()

    lp = LeakagePredictor()
    kw = {k: getattr(a, k) for k in ("vd", "vg", "vs", "vb", "vdd", "temp", "w",
                                     "vth0", "toxe", "ndep", "u0", "xj")}
    r = lp.predict(a.device, state=a.state, **kw).iloc[0]
    print(f"case: {r.case}   model: {lp.info(r.case)['algorithm']}   "
          f"{'(EXTRAPOLATED: outside training ranges)' if r.extrapolated else ''}")
    print("current (A, positive = into device)   ML prediction" + ("        SPICE      rel.err" if a.spice else ""))
    sp = None
    if a.spice:
        import pandas as pd
        from leakage.transistor import simulate_transistors
        m = lp.model(r.case)
        row = {**{k: kw[k] for k in ("vdd", "temp", "w", "vd", "vs", "vg", "vb")},
               **{p: (kw[p] if kw[p] is not None else m.nominal[p]) for p in m.nominal}}
        sp = simulate_transistors(lp.cfg, r.case, pd.DataFrame([row])).iloc[0]
    for t in ("id", "ig", "is", "ib"):
        line = f"  {t}{'':36s}{r[t]: .4e}"
        if sp is not None:
            line += f"   {sp[t]: .4e}   {abs(r[t] - sp[t]) / max(abs(sp[t]), 1e-15):.2%}"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
