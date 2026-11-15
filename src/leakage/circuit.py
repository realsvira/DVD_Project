"""Part C: the assigned test circuit, a 3-bit carry-lookahead adder (CLA3), built only
from the Part B gates (INV, NAND2, NOR2, AND2, OR2).

Logic (i = 0..2, C0 = Cin)
    Gi = Ai . Bi                      Pi = Ai xor Bi
    C1 = G0 + P0.C0
    C2 = G1 + P1.G0 + P1.P0.C0
    C3 = G2 + P2.G1 + P2.P1.G0 + P2.P1.P0.C0      (= Cout)
    Si = Pi xor Ci
Every carry is computed directly from G/P/C0 (lookahead, no ripple); multi-input AND/OR
terms are trees of 2-input gates. XOR(x, y) = AND2(OR2(x, y), NAND2(x, y)); for the
operand XOR the NAND2(Ai, Bi) is shared with Gi = INV(NAND2(Ai, Bi)).

Inputs: A2..A0, B2..B0 and Cin, all driven by sources at exactly 0 or VDD, so every input
combination is 2^7 = 128 vectors: vector_id = A + 8*B + 64*Cin (0..63 have Cin = 0,
64..127 have Cin = 1). The Boolean model is verified for all 128 combinations.

Transistor level: each gate instance is the Part B topology with identical widths, bodies
(NMOS->0, PMOS->vdd), model cards and the corner's shared NMOS / PMOS process draws.
Leakage: same definition as Part B, I_leak = sum_k V_k(-i(V_k)) / VDD over the VDD rail
and all input sources.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config
from .gates import build_gates, corner_process
from .spice import model_text_with

LOGIC = {
    "INV": lambda a: 1 - a,
    "NAND2": lambda a, b: 1 - (a & b),
    "NOR2": lambda a, b: 1 - (a | b),
    "AND2": lambda a, b: a & b,
    "OR2": lambda a, b: a | b,
}


@dataclass(frozen=True)
class GateInst:
    name: str
    kind: str                 # INV | NAND2 | NOR2 | AND2 | OR2
    inputs: tuple[str, ...]   # nets driving pins a (, b)
    output: str               # net driven by pin out


@dataclass(frozen=True)
class Circuit:
    name: str
    primary_inputs: tuple[str, ...]   # driven by sources (operands and carry-in)
    constants: dict                   # net -> logic value (tied inputs; none for CLA3)
    outputs: tuple[str, ...]
    gates: tuple[GateInst, ...]

    @property
    def nets(self) -> list[str]:
        return list(self.primary_inputs) + list(self.constants) + [g.output for g in self.gates]


def _xor(name: str, x: str, y: str, out: str, nand: str | None = None):
    """XOR = AND2(OR2(x,y), NAND2(x,y)); reuse an existing NAND2 output if given."""
    g = []
    if nand is None:
        nand = f"{name}_nd"
        g.append(GateInst(f"{name}_nd", "NAND2", (x, y), nand))
    g += [GateInst(f"{name}_or", "OR2", (x, y), f"{name}_o"),
          GateInst(f"{name}_and", "AND2", (f"{name}_o", nand), out)]
    return g


def build_cla3() -> Circuit:
    g: list[GateInst] = []
    for i in range(3):
        g.append(GateInst(f"nd{i}", "NAND2", (f"a{i}", f"b{i}"), f"nd{i}"))       # NAND(Ai,Bi)
        g.append(GateInst(f"gen{i}", "INV", (f"nd{i}",), f"g{i}"))                 # Gi
        g += _xor(f"px{i}", f"a{i}", f"b{i}", f"p{i}", nand=f"nd{i}")              # Pi
    # C1 = G0 + P0.C0
    g += [GateInst("c1_t", "AND2", ("p0", "cin"), "c1_t"),
          GateInst("c1", "OR2", ("g0", "c1_t"), "c1")]
    # C2 = G1 + P1.G0 + P1.P0.C0
    g += [GateInst("c2_t1", "AND2", ("p1", "g0"), "c2_t1"),
          GateInst("p10", "AND2", ("p1", "p0"), "p10"),
          GateInst("c2_t2", "AND2", ("p10", "cin"), "c2_t2"),
          GateInst("c2_o", "OR2", ("c2_t1", "c2_t2"), "c2_o"),
          GateInst("c2", "OR2", ("g1", "c2_o"), "c2")]
    # C3 = G2 + P2.G1 + P2.P1.G0 + P2.P1.P0.C0
    g += [GateInst("c3_t1", "AND2", ("p2", "g1"), "c3_t1"),
          GateInst("p21", "AND2", ("p2", "p1"), "p21"),
          GateInst("c3_t2", "AND2", ("p21", "g0"), "c3_t2"),
          GateInst("p210", "AND2", ("p21", "p0"), "p210"),
          GateInst("c3_t3", "AND2", ("p210", "cin"), "c3_t3"),
          GateInst("c3_o1", "OR2", ("c3_t1", "c3_t2"), "c3_o1"),
          GateInst("c3_o2", "OR2", ("c3_o1", "c3_t3"), "c3_o2"),
          GateInst("c3", "OR2", ("g2", "c3_o2"), "c3")]
    # Si = Pi xor Ci  (C0 = cin)
    for i, c in enumerate(("cin", "c1", "c2")):
        g += _xor(f"sx{i}", f"p{i}", c, f"s{i}")
    return Circuit("CLA3", ("a0", "a1", "a2", "b0", "b1", "b2", "cin"), {},
                   ("s0", "s1", "s2", "c3"), tuple(g))


# ===================================================================== logic
N_VECTORS = 128                      # A (3 bits) x B (3 bits) x Cin


def vector_bits(vector_id: int, cin: int | None = None) -> dict[str, int]:
    """vector_id = A + 8*B + 64*Cin (A, B in 0..7, Cin in 0..1); bit 0 = LSB.
    An explicit `cin` overrides the carry-in bit of vector_id."""
    if not 0 <= vector_id < N_VECTORS:
        raise ValueError(f"vector_id {vector_id} outside 0..{N_VECTORS - 1}")
    a, b = vector_id % 8, (vector_id // 8) % 8
    cin = vector_id // 64 if cin is None else int(cin)
    return {**{f"a{i}": (a >> i) & 1 for i in range(3)}, **{f"b{i}": (b >> i) & 1 for i in range(3)},
            "cin": cin}


def evaluate(circ: Circuit, values: dict[str, int]) -> dict[str, int]:
    """Logic value of every net (gates are listed in topological order)."""
    v = {**circ.constants, **values}
    for g in circ.gates:
        v[g.output] = LOGIC[g.kind](*(v[n] for n in g.inputs))
    return v


def gate_patterns(circ: Circuit, values: dict[str, int]) -> list[tuple[str, str, str]]:
    """(instance, kind, input bits as on pins a,b) for every gate."""
    v = evaluate(circ, values)
    return [(g.name, g.kind, "".join(str(v[n]) for n in g.inputs)) for g in circ.gates]


def adder_result(values: dict[str, int]) -> int:
    a = sum(values[f"a{i}"] << i for i in range(3))
    b = sum(values[f"b{i}"] << i for i in range(3))
    return a + b + values.get("cin", 0)


def circuit_sum(v: dict[str, int]) -> int:
    return v["s0"] + 2 * v["s1"] + 4 * v["s2"] + 8 * v["c3"]


# ===================================================================== transistor level
def device_list(cfg: Config, circ: Circuit):
    """Flatten gate instances into transistors: (name, type, d, g, s, b, w, instance)."""
    gates = build_gates(cfg)
    devs = []
    for gi in circ.gates:
        topo = gates[gi.kind]
        pins = {"vdd": "vdd", "0": "0", "out": gi.output}
        pins.update(dict(zip(topo.inputs, gi.inputs)))
        pins.update({n: f"{gi.name}_{n}" for n in topo.internal})
        for d in topo.devices:
            devs.append((f"{gi.name}_{d.name}", d.dev, pins[d.d], pins[d.g], pins[d.s], pins[d.b], d.w, gi.name))
    return devs


def circuit_netlist(cfg: Config, circ: Circuit, values: dict[str, int], corner, title: str | None = None) -> str:
    vdd = float(corner["vdd"])
    lines = [title or f"* {circ.name} vector A={sum(values[f'a{i}'] << i for i in range(3))} "
             f"B={sum(values[f'b{i}'] << i for i in range(3))} cin={values.get('cin', 0)} corner={int(corner['corner_id'])}",
             cfg.spice_options, f".temp {float(corner['temp']):.12g}", f"Vdd vdd 0 dc {vdd:.12g}"]
    drive = {**circ.constants, **values}
    for n in list(circ.primary_inputs) + list(circ.constants):
        lines.append(f"V{n} {n} 0 dc {vdd if drive[n] else 0.0:.12g}")
    for name, dev, d, g, s, b, w, _ in device_list(cfg, circ):
        lines.append(f"M{name} {d} {g} {s} {b} {dev} W={w:.12g} L={cfg.lmin:.12g}")
    cards = model_text_with(cfg.spice_model_file, {"nmos": corner_process(corner, "nmos"),
                                                   "pmos": corner_process(corner, "pmos")})
    return "\n".join(lines + ["* ---- model cards (corner NMOS / PMOS process draws)", cards.rstrip(), ".end", ""])


def source_names(circ: Circuit) -> list[str]:
    return ["vdd"] + list(circ.primary_inputs) + list(circ.constants)


def source_current_vector(net: str) -> str:
    """ngspice vector of the source driving `net` (supply source is `Vdd`, inputs `V<net>`)."""
    return "i(vdd)" if net == "vdd" else f"i(v{net})"


def circuit_vectors(circ: Circuit) -> list[str]:
    nets = [g.output for g in circ.gates]
    return [source_current_vector(n) for n in source_names(circ)] + [f"v({n})" for n in nets]


def leakage_from_sources(circ: Circuit, values: dict[str, int], vdd: float, row) -> dict:
    """I_leak = sum_k V_k * (-i(V_k)) / VDD over the VDD rail and all input sources."""
    drive = {**circ.constants, **values}
    p, i_in = 0.0, {}
    for n in source_names(circ):
        cur = -float(row[source_current_vector(n)])
        volt = vdd if (n == "vdd" or drive.get(n, 0)) else 0.0
        p += volt * cur
        i_in[n] = cur
    return {"p_leak": p, "i_leak": p / vdd, "i_vdd": i_in["vdd"],
            "i_inputs_total": sum(v for k, v in i_in.items() if k != "vdd")}


# ===================================================================== ML composition
def predict_circuit_ml(circ: Circuit, vector_ids, corner_ids, gate_leakage) -> pd.DataFrame:
    """Hierarchical ML leakage: sum over gate instances of the gate model's leakage for the
    instance's logic inputs. `gate_leakage(kind, bits) -> pd.Series indexed by corner_id`
    (e.g. GateLeakageModel.leakage). Uses only the input vector and the corner id; no
    full-circuit SPICE quantity enters."""
    corner_ids = list(corner_ids)
    table = {}                           # (kind, bits) -> np.array over corners
    rows = []
    for vid in vector_ids:
        pats = gate_patterns(circ, vector_bits(vid))
        total = np.zeros(len(corner_ids))
        for _, kind, bits in pats:
            if (kind, bits) not in table:
                table[(kind, bits)] = gate_leakage(kind, bits, corner_ids).to_numpy(float)
            total += table[(kind, bits)]
        for k, c in enumerate(corner_ids):
            rows.append({"vector_id": vid, "corner_id": c, "i_leak_ml": total[k]})
    return pd.DataFrame(rows)
