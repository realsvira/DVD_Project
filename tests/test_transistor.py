import hashlib

import numpy as np
import pandas as pd
import pytest

from leakage.config import load_config
from leakage.simulator import NgspiceSimulator
from leakage.spice import find_ngspice
from leakage.transistor import (CURRENT_COLS, kcl_residual, make_inputs, nominal_process,
                                sample_inputs, simulate_standalone, simulate_transistor,
                                simulate_transistors, transistor_circuit)

CFG = load_config()
NGSPICE = find_ngspice(CFG.ngspice_path)
needs_ngspice = pytest.mark.skipif(NGSPICE is None, reason="ngspice not found")
CASES = list(CFG.cases)


@pytest.fixture(scope="module")
def sim():
    return NgspiceSimulator.from_config(CFG, workers=4, chunk=8)


# ------------------------------------------------------------------ inputs
@pytest.mark.parametrize("case", CASES)
def test_sampling_ranges(case):
    df = sample_inputs(CFG, case, 2000, seed=7)
    c = CFG.cases[case]
    nom = nominal_process(CFG, c.device)
    lo, hi = CFG.vdd_range
    assert df.vdd.between(lo, hi).all()
    assert df.temp.between(*CFG.temp_range).all()
    assert df.w.between(*CFG.w_range).all()
    assert (df.vd >= 0).all() and (df.vd <= df.vdd).all()
    assert (df.vs >= 0).all() and (df.vs <= df.vdd).all()
    assert np.allclose(df.vg, c.vg_frac * df.vdd) and np.allclose(df.vb, c.vb_frac * df.vdd)
    for p in CFG.process_params:
        r = df[p] / nom[p]
        assert r.between(0.9 - 1e-9, 1.1 + 1e-9).all(), p


def test_sampling_is_reproducible():
    a = sample_inputs(CFG, "nmos_off", 50)
    assert a.equals(sample_inputs(CFG, "nmos_off", 50))
    assert not a.equals(sample_inputs(CFG, "pmos_off", 50))


@pytest.mark.parametrize("case,vg,vb", [("nmos_on", 0.95, 0.0), ("nmos_off", 0.0, 0.0),
                                        ("pmos_on", 0.0, 0.95), ("pmos_off", 0.95, 0.95)])
def test_case_rules(case, vg, vb):
    r = make_inputs(CFG, case, vdd=0.95)
    assert (r["vg"], r["vb"]) == (vg, vb)


def test_case_rule_violation_rejected():
    r = make_inputs(CFG, "nmos_off")
    r["vg"] = 0.3
    with pytest.raises(ValueError):
        transistor_circuit(CFG, "nmos_off", r)


def test_unknown_process_param_rejected():
    with pytest.raises(ValueError):
        make_inputs(CFG, "nmos_on", process={"vth": 0.4})


@pytest.mark.parametrize("case", CASES)
def test_netlist_content(case):
    dev = CFG.cases[case].device
    other = "pmos" if dev == "nmos" else "nmos"
    r = make_inputs(CFG, case, process={"toxe": 1.111e-9, "vth0": 0.4321 if dev == "nmos" else -0.4321})
    net = transistor_circuit(CFG, case, r).lower()
    assert ".options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9" in net
    for src in ("vd d 0", "vg g 0", "vs s 0", "vb b 0"):
        assert src in net
    assert f"m1 d g s b {dev} " in net and "l=4.5e-08" in net
    assert f".model  {dev}" in net or f".model {dev}" in net
    assert f".model  {other}" not in net and f".model {other}" not in net
    assert "1.111e-09" in net and "0.4321" in net
    assert ".include" not in net            # self-contained; original file only read


# ------------------------------------------------------------------ simulation
@needs_ngspice
@pytest.mark.parametrize("case", CASES)
def test_batch_equals_standalone_netlist(case, sim):
    """The multi-circuit engine must match one-netlist-per-process exactly."""
    df = sample_inputs(CFG, case, 6, seed=123)
    out = simulate_transistors(CFG, case, df, sim)
    assert (out.status == "ok").all()
    for k in range(len(df)):
        ref = simulate_standalone(CFG, NGSPICE, case, df.iloc[k])
        assert np.allclose(out.loc[k, CURRENT_COLS].to_numpy(float), ref, rtol=1e-9, atol=0)


@needs_ngspice
@pytest.mark.parametrize("case", CASES)
def test_nominal_kcl_and_signs(case, sim):
    r = simulate_transistor(CFG, case, sim)
    assert r.status == "ok"
    cur = r[CURRENT_COLS].to_numpy(float)
    assert np.isfinite(cur).all()
    assert abs(cur.sum()) <= 1e-6 * np.abs(cur).max() + 1e-15   # nominal: tighter than KCL_ABS
    # NMOS: Vd > Vs, current enters the drain. PMOS: Vs > Vd, current leaves the drain.
    assert (cur[0] > 0) if CFG.cases[case].device == "nmos" else (cur[0] < 0)


@needs_ngspice
def test_on_current_far_above_off_current(sim):
    for dev in ("nmos", "pmos"):
        on = abs(simulate_transistor(CFG, f"{dev}_on", sim)["id"])
        off = abs(simulate_transistor(CFG, f"{dev}_off", sim)["id"])
        assert on > 1e3 * off


@needs_ngspice
def test_process_variation_reaches_simulator(sim):
    base = simulate_transistor(CFG, "nmos_off", sim)
    low_vt = simulate_transistor(CFG, "nmos_off", sim, process={"vth0": 0.9 * 0.46893})
    thin = simulate_transistor(CFG, "nmos_on", sim, vd=0.0, process={"toxe": 0.9 * 1.25e-9})
    thick = simulate_transistor(CFG, "nmos_on", sim, vd=0.0, process={"toxe": 1.1 * 1.25e-9})
    assert low_vt["id"] > base["id"]                 # lower Vth -> more sub-threshold leakage
    assert abs(thin["ig"]) > abs(thick["ig"])        # thinner oxide -> more gate tunnelling


@needs_ngspice
def test_failures_are_isolated(sim):
    good = transistor_circuit(CFG, "nmos_off", make_inputs(CFG, "nmos_off"))
    bad_model = good.replace("M1 d g s b nmos", "M1 d g s b nosuchmodel")
    singular = good.replace(".end\n", "Vx d 0 dc 0.3\n.end\n")   # two sources fight on node d
    res = sim.run_op([good, bad_model, good, singular, good], ["i(vd)"])
    assert list(res.status) == ["ok", "failed", "ok", "failed", "ok"]
    assert res.values["i(vd)"].iloc[[0, 2, 4]].notna().all()
    assert res.values["i(vd)"].iloc[[1, 3]].isna().all()
    assert any("model" in m.lower() for m in res.messages[1])
    assert res.messages[3]                                         # reason recorded
    assert res.values["i(vd)"].iloc[0] == res.values["i(vd)"].iloc[4]


@needs_ngspice
def test_original_model_file_untouched(sim):
    before = hashlib.sha256(CFG.spice_model_file.read_bytes()).hexdigest()
    simulate_transistors(CFG, "pmos_on", sample_inputs(CFG, "pmos_on", 4, seed=1), sim)
    assert hashlib.sha256(CFG.spice_model_file.read_bytes()).hexdigest() == before
    assert before == CFG.spice_model_sha256


def test_kcl_residual():
    df = pd.DataFrame({"id": [1e-9], "ig": [-2e-12], "is": [-0.998e-9], "ib": [0.0]})
    assert abs(kcl_residual(df).iloc[0]) < 1e-24
