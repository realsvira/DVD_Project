"""Part A: single-transistor netlist generator and simulation API.

Cases (spec Section 4), with VDD the sampled supply:
    nmos_on : Vg = VDD, Vb = 0        nmos_off: Vg = 0,   Vb = 0
    pmos_on : Vg = 0,   Vb = VDD      pmos_off: Vg = VDD, Vb = VDD
Vd and Vs are free inputs (0..VDD in the dataset).

Every circuit is self-contained: the device's own model card (`.model nmos`
or `.model pmos` from the assigned file) is copied into the netlist with the
sample's vth0/toxe/ndep/u0/xj written in. The other device's card is not
included, and the supplied .pm file is only ever read.

Sign convention: every terminal current is positive when it flows INTO the
device terminal (I_X = -i(VX), see spice.py).

Public API
----------
    nominal_process(cfg, device)                      -> {param: value}
    make_inputs(cfg, case, vdd=, temp=, w=, vd=, vs=, process=)  -> one sample (dict)
    sample_inputs(cfg, case, n, seed=)                -> DataFrame of random samples
    transistor_circuit(cfg, case, row)                -> netlist text (no .control)
    example_netlist(cfg, case, row)                   -> stand-alone runnable netlist
    simulate_transistors(cfg, case, df, sim=)         -> df + id, ig, is, ib, status, messages
    simulate_transistor(cfg, case, sim=, **inputs)    -> one result (Series)
    kcl_residual(df)                                  -> id + ig + is + ib
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import Config
from .simulator import NgspiceSimulator
from .spice import model_text_with, parse_prints, read_model_nominals, run_ngspice

INPUT_COLS = ["vdd", "temp", "w", "vd", "vs", "vg", "vb"]
CURRENT_COLS = ["id", "ig", "is", "ib"]
TERMS = ["d", "g", "s", "b"]
VECTORS = [f"i(v{t})" for t in TERMS]


# ===================================================================== inputs
def nominal_process(cfg: Config, device: str) -> dict[str, float]:
    return read_model_nominals(cfg.spice_model_file, cfg.process_params)[device]


def case_seed(cfg: Config, case: str) -> int:
    """Fixed, distinct seed per case (order in project.toml)."""
    return cfg.seeds["data"] + 100 * list(cfg.cases).index(case)


def make_inputs(cfg: Config, case: str, *, vdd: float | None = None, temp: float = 27.0,
                w: float | None = None, vd: float | None = None, vs: float | None = None,
                process: dict[str, float] | None = None) -> dict:
    """One deterministic sample. Defaults: nominal VDD and process, 27 C, W = 2*Lmin,
    full Vds across the device (NMOS: Vd=VDD, Vs=0; PMOS: Vd=0, Vs=VDD).
    Vg and Vb always follow the case rule. `process` may override any subset."""
    c = cfg.cases[case]
    vdd = cfg.vdd_nominal if vdd is None else vdd
    if vd is None:
        vd = vdd if c.device == "nmos" else 0.0
    if vs is None:
        vs = 0.0 if c.device == "nmos" else vdd
    proc = nominal_process(cfg, c.device)
    unknown = set(process or {}) - set(proc)
    if unknown:
        raise ValueError(f"unknown process parameters {unknown}")
    proc.update(process or {})
    return {"vdd": vdd, "temp": temp, "w": 2 * cfg.lmin if w is None else w,
            "vd": vd, "vs": vs, "vg": c.vg_frac * vdd, "vb": c.vb_frac * vdd, **proc}


def sample_inputs(cfg: Config, case: str, n: int, seed: int | None = None) -> pd.DataFrame:
    """n independent random samples (spec Section 4). VDD is drawn first because
    the Vd/Vs ranges depend on it; process params are drawn for this case's device."""
    c = cfg.cases[case]
    nom = nominal_process(cfg, c.device)
    rng = np.random.default_rng(case_seed(cfg, case) if seed is None else seed)
    vdd = rng.uniform(*cfg.vdd_range, n)
    df = pd.DataFrame({
        "vdd": vdd,
        "temp": rng.uniform(*cfg.temp_range, n),
        "w": rng.uniform(*cfg.w_range, n),
        "vd": rng.uniform(0.0, 1.0, n) * vdd,
        "vs": rng.uniform(0.0, 1.0, n) * vdd,
        "vg": c.vg_frac * vdd,
        "vb": c.vb_frac * vdd,
    })
    tol = cfg.process_tolerance
    for p in cfg.process_params:
        df[p] = nom[p] * rng.uniform(1 - tol, 1 + tol, n)
    # round to the 12 significant digits written into the netlist: CSV == simulated
    return df.map(lambda x: float(f"{x:.12g}"))


# ===================================================================== netlists
def _check_row(cfg: Config, case: str, row) -> None:
    c = cfg.cases[case]
    for col, frac in (("vg", c.vg_frac), ("vb", c.vb_frac)):
        if not math.isclose(float(row[col]), frac * float(row["vdd"]), rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"{case}: {col}={row[col]} violates the case rule {frac}*VDD")
    if float(row["w"]) <= 0:
        raise ValueError("width must be positive")


def transistor_circuit(cfg: Config, case: str, row, title: str | None = None) -> str:
    """Self-contained single-FET circuit (no .control), ending in .end."""
    _check_row(cfg, case, row)
    dev = cfg.cases[case].device
    card = model_text_with(cfg.spice_model_file,
                           {dev: {p: float(row[p]) for p in cfg.process_params}}, only=dev)
    g = lambda k: f"{float(row[k]):.12g}"
    return (f"{title or f'* {case} single transistor'}\n"
            f"{cfg.spice_options}\n"
            f".temp {g('temp')}\n"
            f"M1 d g s b {dev} W={g('w')} L={cfg.lmin:.12g}\n"
            f"Vd d 0 dc {g('vd')}\n"
            f"Vg g 0 dc {g('vg')}\n"
            f"Vs s 0 dc {g('vs')}\n"
            f"Vb b 0 dc {g('vb')}\n"
            f"* ---- {dev} model card ({cfg.spice_model_file.name}) with this sample's process values\n"
            f"{card}"
            ".end\n")


def example_netlist(cfg: Config, case: str, row) -> str:
    """Stand-alone, documented netlist (deliverable 2). Run: ngspice_con -b <file>."""
    c = cfg.cases[case]
    head = "\n".join([
        f"* {case.upper()}: single-transistor leakage characterisation",
        f"* Technology: {cfg.model_name} ({cfg.spice_model_file.name}), L = {cfg.lmin*1e9:.0f} nm",
        f"* Case rule : Vg = {c.vg_frac:g}*VDD, Vb = {c.vb_frac:g}*VDD; Vd, Vs free in [0, VDD]",
        f"* Sample    : VDD={float(row['vdd']):.6g} V  T={float(row['temp']):.6g} C  "
        f"W={float(row['w'])*1e9:.6g} nm  Vd={float(row['vd']):.6g} V  Vs={float(row['vs']):.6g} V",
        "* Process   : " + "  ".join(f"{p}={float(row[p]):.6g}" for p in cfg.process_params),
        "* Each terminal has its own voltage source to ground.",
        "* Sign convention: current INTO device terminal X = -i(VX).",
    ])
    body = transistor_circuit(cfg, case, row, title=head).rsplit(".end", 1)[0]
    return body + (".control\nset numdgt=12\nop\nprint i(vd) i(vg) i(vs) i(vb)\n"
                   ".endc\n.end\n")


# ===================================================================== simulation
def simulate_transistors(cfg: Config, case: str, df: pd.DataFrame,
                         sim: NgspiceSimulator | None = None, progress=None) -> pd.DataFrame:
    """Simulate every row. Returns a copy of df with id, ig, is, ib (A, positive into
    the device), `status` ('ok'/'failed') and `messages` (simulator warnings/errors)."""
    sim = sim or NgspiceSimulator.from_config(cfg)
    circuits = [transistor_circuit(cfg, case, r) for r in df.to_dict("records")]
    res = sim.run_op(circuits, VECTORS, progress=progress)
    out = df.reset_index(drop=True).copy()
    for col, vec in zip(CURRENT_COLS, VECTORS):
        out[col] = -res.values[vec].to_numpy()
    out["status"] = res.status.to_numpy()
    out["messages"] = [" | ".join(m) for m in res.messages]
    return out


def simulate_transistor(cfg: Config, case: str, sim: NgspiceSimulator | None = None,
                        **inputs) -> pd.Series:
    """Simulate one sample built with make_inputs(**inputs)."""
    row = make_inputs(cfg, case, **inputs)
    return simulate_transistors(cfg, case, pd.DataFrame([row]), sim).iloc[0]


def kcl_residual(df: pd.DataFrame) -> pd.Series:
    """Sum of the four terminal currents (should be ~0 A)."""
    return df[CURRENT_COLS].sum(axis=1)


# Kirchhoff acceptance: |Id+Ig+Is+Ib| <= KCL_REL * max|I| + KCL_ABS.
# KCL_ABS: with ngspice 44.2 / BSIM4 a residual of ~1.2e-15 A appears when all
# four currents are <= ~1e-12 A (e.g. Vd = Vs). It does not change with gmin,
# reltol or vntol (measured), i.e. it is a fixed non-conservation inside the
# device model, not a solver-tolerance effect; 10 fA bounds it with margin.
KCL_REL = 1e-6
KCL_ABS = 1e-14


def kcl_ok(df: pd.DataFrame) -> pd.Series:
    scale = df[CURRENT_COLS].abs().max(axis=1)
    return kcl_residual(df).abs() <= KCL_REL * scale + KCL_ABS


def simulate_standalone(cfg: Config, exe, case: str, row) -> np.ndarray:
    """Reference path: run the stand-alone example netlist in its own ngspice
    process and parse its printout. Used only to cross-check the batch engine."""
    out = run_ngspice(exe, example_netlist(cfg, case, row), timeout=120)
    v = parse_prints(out)[None]
    return np.array([-v.get(f"i(v{t})", np.nan) for t in TERMS])
