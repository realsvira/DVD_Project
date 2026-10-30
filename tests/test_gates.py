import numpy as np
import pandas as pd
import pytest

from leakage.config import load_config
from leakage.gates import (CORNER_COLUMNS, build_gates, expected_logic, gate_circuit, sample_corners,
                           simulate_gate)
from leakage.simulator import NgspiceSimulator
from leakage.spice import find_ngspice
from leakage.transistor import nominal_process

CFG = load_config()
GATES = build_gates(CFG)
needs_ngspice = pytest.mark.skipif(find_ngspice(CFG.ngspice_path) is None, reason="ngspice not found")


def test_gate_set_and_cases():
    assert list(GATES) == ["INV", "NAND2", "NOR2", "AND2", "OR2"]
    assert sum(len(g.patterns) for g in GATES.values()) == 18
    assert [len(g.devices) for g in GATES.values()] == [2, 4, 4, 6, 6]


def test_topology_bodies_and_widths():
    for g in GATES.values():
        for d in g.devices:
            assert d.b == ("0" if d.dev == "nmos" else "vdd")
            assert CFG.w_range[0] <= d.w <= CFG.w_range[1] + 1e-15
    nand = {d.name: d for d in GATES["NAND2"].devices}
    assert (nand["mna"].d, nand["mna"].s, nand["mnb"].d, nand["mnb"].s) == ("out", "x", "x", "0")
    nor = {d.name: d for d in GATES["NOR2"].devices}
    assert (nor["mpa"].s, nor["mpa"].d, nor["mpb"].s, nor["mpb"].d) == ("vdd", "y", "y", "out")
    and2 = {d.name: d for d in GATES["AND2"].devices}
    assert and2["mna"].d == "n1" and and2["mpi"].g == "n1" and and2["mni"].d == "out"


def test_expected_logic():
    assert expected_logic("INV", "1") == {"out": 0}
    assert expected_logic("NAND2", "11") == {"out": 0} and expected_logic("NAND2", "10") == {"out": 1}
    assert expected_logic("NOR2", "00") == {"out": 1} and expected_logic("NOR2", "01") == {"out": 0}
    assert expected_logic("AND2", "11") == {"n1": 0, "out": 1}
    assert expected_logic("OR2", "00") == {"n1": 1, "out": 0}


def test_corners_reproducible_and_in_range():
    c = sample_corners(CFG)
    assert len(c) == 200 and list(c.columns) == CORNER_COLUMNS
    pd.testing.assert_frame_equal(c, sample_corners(CFG))
    pd.testing.assert_frame_equal(c.head(10), sample_corners(CFG, n=10))     # corner k independent of n
    lo, hi = CFG.vdd_range
    assert c.vdd.between(lo, hi).all() and c.temp.between(*CFG.temp_range).all()
    for d, pre in (("nmos", "n_"), ("pmos", "p_")):
        for p, v in nominal_process(CFG, d).items():
            assert (c[pre + p] / v).between(0.9 - 1e-9, 1.1 + 1e-9).all()
    assert not np.allclose(c.n_vth0 / nominal_process(CFG, "nmos")["vth0"],
                           c.p_vth0 / nominal_process(CFG, "pmos")["vth0"])   # separate draws


def test_netlist_has_options_inputs_and_both_cards():
    c = sample_corners(CFG, n=1).iloc[0]
    net = gate_circuit(CFG, GATES["NAND2"], "10", c).lower()
    assert CFG.spice_options in net
    assert f"va a 0 dc {float(c.vdd):.12g}" in net and "vb b 0 dc 0" in net
    assert net.count(".model") == 2                         # one NMOS + one PMOS card
    assert net.count("\nmm") == 4                          # 4 transistors in NAND2
    assert f"{float(c.n_vth0):.10g}" in net and f"{float(c.p_toxe):.10g}" in net


@needs_ngspice
def test_all_gates_truth_table_and_power_balance_at_corner0():
    sim = NgspiceSimulator.from_config(CFG, workers=4, chunk=10)
    c = sample_corners(CFG, n=1)
    for g in GATES.values():
        s, d, _ = simulate_gate(CFG, g, c, sim)
        assert (s.status == "ok").all() and s.logic_ok.all(), g.name
        assert (s.power_balance_rel < 1e-6).all() and (s.i_leak > 0).all()
        assert len(d) == len(s) * len(g.devices)
