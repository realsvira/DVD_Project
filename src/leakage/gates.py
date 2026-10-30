"""Part B: static CMOS gates, PVT corners and transistor-level SPICE references.

Gate topologies are plain data (`Gate` with a list of `Device`s) so the same
description drives (a) the SPICE netlists here and (b) the ML gate model in
Part B step 4, which needs every transistor's terminal nodes.

Nodes: vdd (supply rail), 0 (ground), a / b (inputs), out (output), and
  NAND2: x  = node between the series NMOS (MNA on top, MNB at ground)
  NOR2 : y  = node between the series PMOS (MPA at VDD, MPB at the output)
  AND2 : x and n1 (NAND2 output, inverter input), out
  OR2  : y and n1 (NOR2 output, inverter input), out
Bodies: NMOS -> 0, PMOS -> vdd.

Sizing (config [gates]): unit inverter Wn = 2 Lmin, Wp = 4 Lmin; a series stack of
k devices uses k x unit width, parallel devices unit width (equal-drive sizing).

Current probes: every device terminal is connected through its own 0 V source
`V<dev>_<t>` from the circuit node (+) to the device pin (-), so
    I_into_device_terminal = +i(V<dev>_<t>)
(note: + sign here, because the source's + side is the circuit node). A 0 V
source changes nothing electrically.

Leakage definition (positive magnitude)
  Each source V_k (VDD rail, inputs) delivers current  I_k = -i(V_k)  into the
  circuit (ngspice reports i(V) flowing + -> - through the source).
  P_leak = sum_k V_k * I_k   (total static power; inputs at 0 V deliver no power)
  I_leak = P_leak / VDD      <- the reference "gate leakage current" (A, > 0)
  It includes sub-threshold, gate-tunnelling (also from inputs held at VDD) and
  junction/body leakage. The VDD-rail current alone, I_vdd = -i(Vdd), is recorded too.
  By Tellegen's theorem P_leak also equals sum over devices and terminals of
  V_node * I_into_terminal, which is how the ML side (step 4) computes it.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

from .config import Config
from .simulator import NgspiceSimulator
from .spice import model_text_with
from .transistor import nominal_process

TERMS = ("d", "g", "s", "b")


@dataclass(frozen=True)
class Device:
    name: str        # e.g. "mna"
    dev: str         # "nmos" | "pmos"
    d: str
    g: str
    s: str
    b: str
    w: float         # m

    def node(self, t: str) -> str:
        return getattr(self, t)


@dataclass(frozen=True)
class Gate:
    name: str
    inputs: tuple[str, ...]
    internal: tuple[str, ...]       # recorded internal nodes (besides out)
    devices: tuple[Device, ...]

    @property
    def patterns(self) -> list[str]:
        return ["".join(p) for p in product("01", repeat=len(self.inputs))]

    @property
    def nodes(self) -> tuple[str, ...]:
        return self.internal + ("out",)


def _logic(gate: str, bits: str) -> dict[str, int]:
    """Ideal logic value of every recorded logic node for an input pattern."""
    a = int(bits[0])
    b = int(bits[1]) if len(bits) > 1 else None
    if gate == "INV":
        return {"out": 1 - a}
    nand, nor = 1 - (a & b), 1 - (a | b)
    return {"NAND2": {"out": nand}, "NOR2": {"out": nor},
            "AND2": {"n1": nand, "out": 1 - nand}, "OR2": {"n1": nor, "out": 1 - nor}}[gate]


def expected_logic(gate: str, bits: str) -> dict[str, int]:
    return _logic(gate, bits)


def build_gates(cfg: Config) -> dict[str, Gate]:
    g = cfg.raw["gates"]
    wn, wp = g["wn_unit_lmin"] * cfg.lmin, g["wp_unit_lmin"] * cfg.lmin
    N = lambda n, d, gg, s, w: Device(n, "nmos", d, gg, s, "0", w)          # noqa: E731
    P = lambda n, d, gg, s, w: Device(n, "pmos", d, gg, s, "vdd", w)        # noqa: E731

    def inv(prefix, a, out):
        return (P(f"mp{prefix}", out, a, "vdd", wp), N(f"mn{prefix}", out, a, "0", wn))

    def nand(out, x):
        return (P("mpa", out, "a", "vdd", wp), P("mpb", out, "b", "vdd", wp),
                N("mna", out, "a", x, 2 * wn), N("mnb", x, "b", "0", 2 * wn))

    def nor(out, y):
        return (P("mpa", y, "a", "vdd", 2 * wp), P("mpb", out, "b", y, 2 * wp),
                N("mna", out, "a", "0", wn), N("mnb", out, "b", "0", wn))

    gates = [
        Gate("INV", ("a",), (), inv("", "a", "out")),
        Gate("NAND2", ("a", "b"), ("x",), nand("out", "x")),
        Gate("NOR2", ("a", "b"), ("y",), nor("out", "y")),
        Gate("AND2", ("a", "b"), ("x", "n1"), nand("n1", "x") + inv("i", "n1", "out")),
        Gate("OR2", ("a", "b"), ("y", "n1"), nor("n1", "y") + inv("i", "n1", "out")),
    ]
    out = {gt.name: gt for gt in gates}
    for gt in gates:
        for dv in gt.devices:
            if not (cfg.w_range[0] - 1e-15 <= dv.w <= cfg.w_range[1] + 1e-15):
                raise ValueError(f"{gt.name}.{dv.name}: W={dv.w} outside the ML training range")
    return out


# ===================================================================== corners
CORNER_COLUMNS = ["corner_id", "vdd", "temp"] + [f"n_{p}" for p in ("vth0", "toxe", "ndep", "u0", "xj")] \
                 + [f"p_{p}" for p in ("vth0", "toxe", "ndep", "u0", "xj")]


def sample_corners(cfg: Config, n: int | None = None, seed: int | None = None) -> pd.DataFrame:
    """n random PVT corners (spec Section 4): VDD first, then T, then one NMOS draw of
    the five process parameters and one separate PMOS draw. Corner k is reproducible
    and independent of n (one RNG per corner)."""
    n = n or int(cfg.raw["gates"]["n_corners"])
    seed = cfg.seeds["corners"] if seed is None else seed
    nom = {d: nominal_process(cfg, d) for d in ("nmos", "pmos")}
    tol = cfg.process_tolerance
    rows = []
    for k in range(n):
        rng = np.random.default_rng([seed, k])
        r = {"corner_id": k, "vdd": rng.uniform(*cfg.vdd_range), "temp": rng.uniform(*cfg.temp_range)}
        for d, pre in (("nmos", "n_"), ("pmos", "p_")):
            for p in cfg.process_params:
                r[pre + p] = nom[d][p] * rng.uniform(1 - tol, 1 + tol)
        rows.append(r)
    df = pd.DataFrame(rows)[CORNER_COLUMNS]
    num = [c for c in df.columns if c != "corner_id"]
    df[num] = df[num].map(lambda x: float(f"{x:.12g}"))
    return df


def corner_process(corner, device: str) -> dict[str, float]:
    pre = "n_" if device == "nmos" else "p_"
    return {p: float(corner[pre + p]) for p in ("vth0", "toxe", "ndep", "u0", "xj")}


# ===================================================================== netlists
def probe(dev: Device, t: str) -> str:
    return f"v{dev.name}_{t}"


def gate_circuit(cfg: Config, gate: Gate, bits: str, corner, title: str | None = None) -> str:
    vdd = float(corner["vdd"])
    lines = [title or f"* {gate.name} inputs={bits} corner={int(corner['corner_id'])}",
             cfg.spice_options,
             f".temp {float(corner['temp']):.12g}",
             f"Vdd vdd 0 dc {vdd:.12g}"]
    for name, bit in zip(gate.inputs, bits):
        lines.append(f"V{name} {name} 0 dc {vdd if bit == '1' else 0.0:.12g}")
    lines.append("* devices; each terminal reaches its node through a 0 V current probe")
    for dv in gate.devices:
        pins = " ".join(f"{dv.name}_{t}" for t in TERMS)
        lines.append(f"M{dv.name} {pins} {dv.dev} W={dv.w:.12g} L={cfg.lmin:.12g}")
        for t in TERMS:
            lines.append(f"{probe(dv, t).upper()} {dv.node(t)} {dv.name}_{t} dc 0")
    cards = model_text_with(cfg.spice_model_file, {"nmos": corner_process(corner, "nmos"),
                                                   "pmos": corner_process(corner, "pmos")})
    lines += ["* ---- model cards with this corner's NMOS and PMOS process draws", cards.rstrip(), ".end", ""]
    return "\n".join(lines)


def example_netlist(cfg: Config, gate: Gate, bits: str, corner) -> str:
    head = "\n".join([
        f"* {gate.name} transistor-level leakage reference, inputs {''.join(gate.inputs).upper()}={bits}",
        f"* Technology {cfg.model_name}, L = {cfg.lmin*1e9:.0f} nm; corner {int(corner['corner_id'])}: "
        f"VDD={float(corner['vdd']):.6g} V, T={float(corner['temp']):.6g} C",
        "* Devices: " + ", ".join(f"{d.name}({d.dev}, W={d.w*1e9:.0f}n, d={d.d} g={d.g} s={d.s} b={d.b})"
                                  for d in gate.devices),
        "* Leakage: I_leak = sum_k V_k * (-i(V_k)) / VDD over the sources Vdd, Va, Vb",
        "* Probe sign: current INTO device terminal = +i(V<dev>_<t>)"])
    body = gate_circuit(cfg, gate, bits, corner, title=head).rsplit(".end", 1)[0]
    vec = " ".join([f"v({n})" for n in gate.nodes] + ["i(vdd)"] + [f"i(v{n})" for n in gate.inputs])
    return body + f".control\nset numdgt=10\nop\nprint {vec}\n.endc\n.end\n"


def gate_vectors(gate: Gate) -> list[str]:
    v = ["i(vdd)"] + [f"i(v{n})" for n in gate.inputs] + [f"v({n})" for n in gate.nodes]
    v += [f"i({probe(dv, t)})" for dv in gate.devices for t in TERMS]
    return v


# ===================================================================== simulation
def simulate_gate(cfg: Config, gate: Gate, corners: pd.DataFrame, sim: NgspiceSimulator,
                  patterns: list[str] | None = None):
    """Simulate every (pattern, corner). Returns (summary, devices):
    summary: one row per simulation (leakage, node voltages, checks);
    devices: one row per simulation x device with terminal voltages and SPICE currents."""
    patterns = patterns or gate.patterns
    jobs = [(bits, c) for bits in patterns for _, c in corners.iterrows()]
    vecs = gate_vectors(gate)
    res = sim.run_op([gate_circuit(cfg, gate, bits, c) for bits, c in jobs], vecs)
    tol = float(cfg.raw["gates"].get("logic_tolerance", 0.05))
    summary, devices = [], []
    for k, (bits, c) in enumerate(jobs):
        v = res.values.iloc[k]
        vdd = float(c.vdd)
        volt = {"0": 0.0, "vdd": vdd, **{n: (vdd if b == "1" else 0.0) for n, b in zip(gate.inputs, bits)},
                **{n: float(v[f"v({n})"]) for n in gate.nodes}}
        src = {"vdd": -float(v["i(vdd)"]), **{n: -float(v[f"i(v{n})"]) for n in gate.inputs}}
        p_src = sum(volt[n] * i for n, i in src.items())
        p_dev = 0.0
        for dv in gate.devices:
            cur = {t: float(v[f"i({probe(dv, t)})"]) for t in TERMS}
            p_dev += sum(volt[dv.node(t)] * cur[t] for t in TERMS)
            devices.append({"gate": gate.name, "inputs": bits, "corner_id": int(c.corner_id),
                            "device": dv.name, "type": dv.dev, "w": dv.w,
                            **{f"node_{t}": dv.node(t) for t in TERMS},
                            **{f"v{t}": volt[dv.node(t)] for t in TERMS},
                            **{f"i{t}": cur[t] for t in TERMS}})
        exp = expected_logic(gate.name, bits)
        logic_ok = all(abs(volt[n] - val * vdd) <= tol * vdd for n, val in exp.items())
        row = {"gate": gate.name, "inputs": bits, "corner_id": int(c.corner_id),
               **{col: c[col] for col in CORNER_COLUMNS if col != "corner_id"},
               "i_leak": p_src / vdd, "p_leak": p_src, "i_vdd": src["vdd"],
               **{f"i_in_{n}": src[n] for n in gate.inputs},
               **{f"v_{n}": volt[n] for n in gate.nodes},
               **{f"expected_{n}": val for n, val in exp.items()},
               "logic_ok": logic_ok,
               "power_balance_rel": abs(p_src - p_dev) / max(abs(p_src), 1e-30),
               "status": res.status.iloc[k], "messages": " | ".join(res.messages[k])}
        summary.append(row)
    return pd.DataFrame(summary), pd.DataFrame(devices), res
