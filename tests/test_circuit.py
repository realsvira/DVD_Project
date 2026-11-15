import itertools

import numpy as np
import pandas as pd
import pytest

from leakage.circuit import (adder_result, build_cla3, circuit_netlist, circuit_sum, device_list,
                             evaluate, gate_patterns, predict_circuit_ml, vector_bits)
from leakage.config import load_config
from leakage.gates import sample_corners

CFG = load_config()
CLA = build_cla3()


def test_exhaustive_truth_table_cin0_and_cin1():
    """All 64 operand pairs for Cin = 0 (the simulated case) and Cin = 1 (logic check)."""
    for cin in (0, 1):
        for vid in range(64):
            v = vector_bits(vid, cin)
            assert circuit_sum(evaluate(CLA, v)) == adder_result(v), (vid, cin)


def test_carries_are_lookahead_and_match_definitions():
    for vid, cin in itertools.product(range(64), (0, 1)):
        v = evaluate(CLA, vector_bits(vid, cin))
        g = [v[f"a{i}"] & v[f"b{i}"] for i in range(3)]
        p = [v[f"a{i}"] ^ v[f"b{i}"] for i in range(3)]
        assert [v[f"g{i}"] for i in range(3)] == g and [v[f"p{i}"] for i in range(3)] == p
        c1 = g[0] | (p[0] & cin)
        c2 = g[1] | (p[1] & g[0]) | (p[1] & p[0] & cin)
        c3 = g[2] | (p[2] & g[1]) | (p[2] & p[1] & g[0]) | (p[2] & p[1] & p[0] & cin)
        assert (v["c1"], v["c2"], v["c3"]) == (c1, c2, c3)
    # lookahead: C2/C3 are not computed from C1/C2
    drivers = {g.output: g for g in CLA.gates}

    def cone(net, seen=None):
        seen = set() if seen is None else seen
        if net in drivers:
            for n in drivers[net].inputs:
                if n not in seen:
                    seen.add(n)
                    cone(n, seen)
        return seen
    assert "c1" not in cone("c2") and "c1" not in cone("c3") and "c2" not in cone("c3")


def test_only_supported_gates_and_counts():
    kinds = [g.kind for g in CLA.gates]
    assert set(kinds) <= {"INV", "NAND2", "NOR2", "AND2", "OR2"}
    assert len(CLA.gates) == 36 and len({g.name for g in CLA.gates}) == 36
    assert len({g.output for g in CLA.gates}) == 36            # every net has exactly one driver
    assert len(device_list(CFG, CLA)) == 192


def test_gate_patterns_cover_known_cases():
    pats = {name: (kind, bits) for name, kind, bits in gate_patterns(CLA, vector_bits(63))}  # A=7, B=7
    assert pats["nd0"] == ("NAND2", "11") and pats["gen0"] == ("INV", "0")
    assert pats["c1_t"][1][1] == "0"                           # cin tied low


def test_netlist_structure():
    c = sample_corners(CFG, n=1).iloc[0]
    net = circuit_netlist(CFG, CLA, vector_bits(5), c).lower()
    assert CFG.spice_options in net and net.count(".model") == 2
    assert net.count("\nm") == 192
    assert "vcin cin 0 dc 0" in net
    assert f"va0 a0 0 dc {float(c.vdd):.12g}" in net and "va1 a1 0 dc 0" in net   # A=5 -> a0=1, a1=0


@pytest.mark.skipif(not (CFG.data / "gates" / "gate_spice_results.csv").is_file(), reason="Part B3 not run")
def test_cla_ml_equals_sum_of_part_b_gate_predictions():
    """The CLA estimate must equal the sum of the gate-model predictions (same models)."""
    from leakage.gate_ml import GateLeakageModel, predict_gate_records
    gm = GateLeakageModel()
    refs = pd.read_csv(CFG.data / "gates" / "gate_spice_results.csv", dtype={"inputs": str})
    g, _ = predict_gate_records(refs[refs.corner_id.isin([0, 7])], gm.predictor)
    b4 = g.set_index(["gate", "inputs", "corner_id"]).i_leak
    out = predict_circuit_ml(CLA, [5 + 8 * 3], [0, 7], gm.leakage)
    for c in (0, 7):
        expect = sum(b4.loc[(k, b, c)] for _, k, b in gate_patterns(CLA, vector_bits(5 + 8 * 3)))
        assert out[out.corner_id == c].i_leak_ml.iloc[0] == pytest.approx(expect, rel=1e-12)


def test_ml_composition_sums_gate_models():
    calls = []

    def fake(kind, bits, corners):
        calls.append((kind, bits))
        return pd.Series(np.full(len(corners), 1e-9), index=corners)
    out = predict_circuit_ml(CLA, [0, 63], [0, 1, 2], fake)
    assert len(out) == 6 and np.allclose(out.i_leak_ml, 36e-9)
    assert len(calls) == len(set(calls))                      # each (kind, bits) evaluated once


def test_all_128_input_combinations_include_cin():
    """vector_id = A + 8*B + 64*Cin covers every input combination; 0..63 keep Cin = 0."""
    seen = set()
    for vid in range(128):
        v = vector_bits(vid)
        assert v["cin"] == vid // 64
        assert circuit_sum(evaluate(CLA, v)) == adder_result(v)
        seen.add(tuple(v[k] for k in ("a0", "a1", "a2", "b0", "b1", "b2", "cin")))
    assert len(seen) == 128
    with pytest.raises(ValueError):
        vector_bits(128)


def test_cin1_netlist_drives_cin_high():
    c = sample_corners(CFG, n=1).iloc[0]
    net = circuit_netlist(CFG, CLA, vector_bits(64 + 5), c).lower()
    assert f"vcin cin 0 dc {float(c.vdd):.12g}" in net
    assert circuit_netlist(CFG, CLA, vector_bits(5), c) != circuit_netlist(CFG, CLA, vector_bits(69), c)
