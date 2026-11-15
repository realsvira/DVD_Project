"""Environment check: Python, packages, config, SPICE model file and ngspice.

Run from the project root (with the virtual environment active):
    python scripts/check_environment.py

Exit code 0 = every required check passed. A summary is also written to
reports/environment_check.txt.
"""
from __future__ import annotations

import hashlib
import importlib
import math
import platform
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RESULTS: list[tuple[str, str, str]] = []   # (status, check, detail)


def record(ok: bool | None, check: str, detail: str = "") -> bool:
    status = "PASS" if ok else ("WARN" if ok is None else "FAIL")
    RESULTS.append((status, check, detail))
    print(f"[{status}] {check}" + (f"  -- {detail}" if detail else ""))
    return bool(ok)


# ------------------------------------------------------------------ 1. Python
def check_python() -> None:
    v = sys.version_info
    record(v >= (3, 11), "Python >= 3.11", f"{platform.python_version()} at {sys.executable}")
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    record(True if in_venv else None, "Running inside a virtual environment",
           sys.prefix if in_venv else "not in a venv (activate .venv first)")


REQUIRED_PKGS = ["numpy", "pandas", "scipy", "sklearn", "matplotlib", "joblib",
                 "lightgbm", "pytest"]


def check_packages() -> None:
    for name in REQUIRED_PKGS:
        try:
            mod = importlib.import_module(name)
            record(True, f"import {name}", getattr(mod, "__version__", "?"))
        except Exception as e:  # noqa: BLE001
            record(False, f"import {name}", f"{type(e).__name__}: {e}")


# ------------------------------------------------------------------ 2. config + model
UNITS = {"vth0": "V", "toxe": "m", "ndep": "cm^-3", "u0": "m^2/(V*s)", "xj": "m"}


def check_config_and_model():
    from leakage.config import load_config
    from leakage.spice import model_info, read_model_nominals
    try:
        cfg = load_config()
    except Exception as e:  # noqa: BLE001
        record(False, "Load config/project.toml", str(e))
        return None, None
    record(True, "Load config/project.toml",
           f"model={cfg.model_name}, VDD={cfg.vdd_nominal} V, Lmin={cfg.lmin*1e9:.0f} nm, "
           f"test circuit={cfg.test_circuit}")

    for d in (cfg.netlists, cfg.data, cfg.trained_models, cfg.results, cfg.figures, cfg.reports):
        d.mkdir(parents=True, exist_ok=True)
    record(True, "Project output folders exist")

    mf = cfg.spice_model_file
    if not record(mf.is_file(), "SPICE model file found", str(mf)):
        return cfg, None
    digest = hashlib.sha256(mf.read_bytes()).hexdigest()
    record(digest == cfg.spice_model_sha256, "Model file SHA-256 matches config",
           digest[:16] + "...")

    info = model_info(mf)
    for dev in ("nmos", "pmos"):
        d = info.get(dev)
        record(d is not None and d["type"] == dev and d["level"] == 54,
               f"Model card '{dev}' present, BSIM4 (level 54)",
               f"level={d['level'] if d else '-'}, version={d['params'].get('version') if d else '-'}")
    try:
        nominals = read_model_nominals(mf, cfg.process_params)
    except ValueError as e:
        record(False, "Process parameters present", str(e))
        return cfg, None
    for dev, vals in nominals.items():
        record(True, f"{dev} nominal process params",
               ", ".join(f"{k}={v:g} {UNITS.get(k, '')}" for k, v in vals.items()))
    return cfg, nominals


# ------------------------------------------------------------------ 3. ngspice
def check_ngspice(cfg, nominals) -> None:
    from leakage.spice import (find_ngspice, model_text_with, ngspice_version,
                               parse_prints, run_ngspice, simulator_errors, spice_path)
    exe = find_ngspice(cfg.ngspice_path)
    if not record(exe is not None, "ngspice executable found",
                  str(exe) if exe else f"not at {cfg.ngspice_path}, not on PATH; set NGSPICE"):
        return
    record(exe.name.lower() == "ngspice_con.exe" or not exe.suffix, "Using console ngspice build",
           exe.name)
    ver = ngspice_version(exe)
    record(ver != "unknown", "ngspice responds to -v", f"ngspice {ver}")
    if nominals is None:
        return

    vdd, w, L = cfg.vdd_nominal, 2 * cfg.lmin, cfg.lmin
    # NMOS OFF (Vg=0, Vb=0) and PMOS OFF (Vg=Vb=VDD), every terminal on its own source
    devices = (f"Mn dn gn sn bn nmos W={w:.4e} L={L:.4e}\n"
               f"Vdn dn 0 dc {vdd}\nVgn gn 0 dc 0\nVsn sn 0 dc 0\nVbn bn 0 dc 0\n"
               f"Mp dp gp sp bp pmos W={w:.4e} L={L:.4e}\n"
               f"Vdp dp 0 dc 0\nVgp gp 0 dc {vdd}\nVsp sp 0 dc {vdd}\nVbp bp 0 dc {vdd}\n")
    control = (".control\nset numdgt=10\nop\n"
               "print i(vdn) i(vgn) i(vsn) i(vbn) i(vdp) i(vgp) i(vsp) i(vbp)\n"
               "showmod mn : vth0 toxe ndep u0 xj\nshowmod mp : vth0 toxe ndep u0 xj\n"
               ".endc\n.end\n")
    net = (f"* environment check: {cfg.model_name} single NMOS/PMOS OFF at 27 C\n"
           f".include {spice_path(cfg.spice_model_file)}\n{cfg.spice_options}\n.temp 27\n"
           f"{devices}{control}")
    t0 = time.perf_counter()
    try:
        out = run_ngspice(exe, net, timeout=120)
    except Exception as e:  # noqa: BLE001
        record(False, "ngspice batch simulation runs", str(e)[-500:])
        return
    record(True, "ngspice batch simulation runs (Python -> ngspice)",
           f"{time.perf_counter() - t0:.2f} s")
    errs = simulator_errors(out)
    record(not errs, "No simulator errors in output", "; ".join(errs[:3]))

    # model parameters as seen by the simulator == values in the file
    seen, cur = {}, None
    for line in out.splitlines():
        m = re.match(r"\s*model\s+(\w+)\s*$", line)
        if m:
            cur = m.group(1).lower()
            seen[cur] = {}
            continue
        m = re.match(r"\s*(vth0|toxe|ndep|u0|xj)\s+(\S+)\s*$", line)
        if m and cur:
            seen[cur][m.group(1)] = float(m.group(2))
    match = all(math.isclose(seen.get(d, {}).get(p, float("nan")), v, rel_tol=1e-6)
                for d, vals in nominals.items() for p, v in vals.items())
    record(match, "Simulator loaded the assigned model (showmod == file)", str(seen) if not match else "")

    vals = parse_prints(out)[None]
    for dev, tag in (("NMOS OFF", "n"), ("PMOS OFF", "p")):
        cur_into = {t: -vals.get(f"i(v{t}{tag})", float("nan")) for t in "dgsb"}
        finite = all(math.isfinite(v) for v in cur_into.values())
        kcl = abs(sum(cur_into.values()))
        scale = max(abs(v) for v in cur_into.values()) if finite else float("nan")
        detail = ", ".join(f"I{t}={v:.3e}" for t, v in cur_into.items()) + f"  |sum|={kcl:.1e} A"
        record(finite and scale > 0 and kcl <= 1e-6 * scale + 1e-15,
               f"{dev}: currents finite and Kirchhoff sum ~ 0", detail)

    # process override path: toxe -10% in a fresh model card must raise gate leakage
    ovr = model_text_with(cfg.spice_model_file,
                          {"nmos": {"toxe": nominals["nmos"]["toxe"] * 0.9}}, only="nmos")
    net2 = (f"* override check\n{cfg.spice_options}\n.temp 27\n"
            f"Mn dn gn sn bn nmos W={w:.4e} L={L:.4e}\n"
            f"Vdn dn 0 dc {vdd}\nVgn gn 0 dc 0\nVsn sn 0 dc 0\nVbn bn 0 dc 0\n"
            f"{ovr}.control\nset numdgt=10\nop\nprint i(vgn)\n.endc\n.end\n")
    ig_thin = abs(parse_prints(run_ngspice(exe, net2, timeout=120))[None].get("i(vgn)", float("nan")))
    ig_nom = abs(vals.get("i(vgn)", float("nan")))
    record(ig_thin > ig_nom, "Model-card override works (toxe -10% -> higher NMOS gate leakage)",
           f"Ig nominal={ig_nom:.3e} A, thin oxide={ig_thin:.3e} A")


def main() -> int:
    print(f"Project root: {ROOT}\n")
    check_python()
    check_packages()
    cfg, nominals = check_config_and_model()
    if cfg is not None:
        check_ngspice(cfg, nominals)

    n_fail = sum(s == "FAIL" for s, _, _ in RESULTS)
    n_warn = sum(s == "WARN" for s, _, _ in RESULTS)
    verdict = "ENVIRONMENT OK" if n_fail == 0 else f"ENVIRONMENT NOT READY ({n_fail} failed)"
    print(f"\n{verdict}  ({len(RESULTS)} checks, {n_warn} warnings)")

    rep = ROOT / "reports" / "environment_check.txt"
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {platform.platform()}\n"
                   + "\n".join(f"[{s}] {c}" + (f"  -- {d}" if d else "") for s, c, d in RESULTS)
                   + f"\n\n{verdict}\n", encoding="utf-8")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
