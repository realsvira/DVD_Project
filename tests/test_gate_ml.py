import numpy as np
import pandas as pd
import pytest

from leakage.config import load_config
from leakage.gate_ml import device_power, gate_leakage, terminal_table
from leakage.gates import build_gates

CFG = load_config()
GATES = build_gates(CFG)
GDIR = CFG.data / "gates"
HAVE_REFS = (GDIR / "gate_spice_results.csv").is_file() and (GDIR / "gate_spice_devices.csv").is_file()
MODELS = all((CFG.trained_models / f"{c}.joblib").is_file() for c in CFG.cases)


def record(gate, bits, **nodes):
    r = {"gate": gate, "inputs": bits, "corner_id": 0, "vdd": 1.0, "temp": 27.0,
         **{f"n_{p}": 1.0 for p in ("vth0", "toxe", "ndep", "u0", "xj")},
         **{f"p_{p}": -1.0 if p == "vth0" else 1.0 for p in ("vth0", "toxe", "ndep", "u0", "xj")}}
    r.update({f"v_{k}": v for k, v in nodes.items()})
    return r


def test_nand2_terminal_mapping_and_states():
    t = terminal_table(pd.DataFrame([record("NAND2", "10", out=1.0, x=0.77)]), GATES).set_index("device")
    mna, mnb, mpa, mpb = (t.loc[n] for n in ("mna", "mnb", "mpa", "mpb"))
    assert (mna.vd, mna.vg, mna.vs, mna.vb) == (1.0, 1.0, 0.77, 0.0) and mna.case == "nmos_on"
    assert (mnb.vd, mnb.vg, mnb.vs, mnb.vb) == (0.77, 0.0, 0.0, 0.0) and mnb.case == "nmos_off"
    assert (mpa.vd, mpa.vg, mpa.vs, mpa.vb) == (1.0, 1.0, 1.0, 1.0) and mpa.case == "pmos_off"
    assert (mpb.vg, mpb.case) == (0.0, "pmos_on")


def test_nor2_series_pmos_mapping():
    t = terminal_table(pd.DataFrame([record("NOR2", "10", out=0.0, y=0.24)]), GATES).set_index("device")
    assert (t.loc["mpa"].vs, t.loc["mpa"].vd, t.loc["mpa"].case) == (1.0, 0.24, "pmos_off")
    assert (t.loc["mpb"].vs, t.loc["mpb"].vd, t.loc["mpb"].case) == (0.24, 0.0, "pmos_on")
    assert t.loc["mna"].case == "nmos_on" and t.loc["mnb"].case == "nmos_off"


def test_internal_gate_node_is_snapped_and_recorded():
    t = terminal_table(pd.DataFrame([record("AND2", "11", out=1.0, x=3e-6, n1=5e-6)]), GATES).set_index("device")
    assert t.loc["mpi"].vg == 0.0 and t.loc["mpi"].vg_node == 5e-6 and t.loc["mpi"].vg_snap == pytest.approx(5e-6)
    assert t.loc["mpi"].case == "pmos_on" and t.loc["mni"].case == "nmos_off"


def test_intermediate_gate_voltage_rejected():
    with pytest.raises(ValueError):
        terminal_table(pd.DataFrame([record("AND2", "11", out=1.0, x=3e-6, n1=0.4)]), GATES)


def test_device_power_equals_tellegen_sum_when_kcl_holds():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"vd": rng.uniform(0, 1, 50), "vg_node": rng.uniform(0, 1, 50),
                       "vs": rng.uniform(0, 1, 50), "vb": rng.uniform(0, 1, 50),
                       "id": rng.normal(size=50), "ig": rng.normal(size=50), "ib": rng.normal(size=50)})
    df["is"] = -(df["id"] + df["ig"] + df["ib"])
    full = df.vd * df["id"] + df.vg_node * df.ig + df.vs * df["is"] + df.vb * df.ib
    assert np.allclose(device_power(df), full, rtol=1e-12, atol=1e-15)


@pytest.mark.skipif(not HAVE_REFS, reason="gate SPICE references not generated")
def test_accounting_with_spice_currents_reproduces_spice_leakage():
    """Feed SPICE's own per-transistor currents through the ML accounting code."""
    dev = pd.read_csv(GDIR / "gate_spice_devices.csv", dtype={"inputs": str})
    ref = pd.read_csv(GDIR / "gate_spice_results.csv", dtype={"inputs": str})
    dev["vg_node"] = dev["vg"]
    dev = dev.merge(ref[["gate", "inputs", "corner_id", "vdd"]], on=["gate", "inputs", "corner_id"])
    g = gate_leakage(dev).merge(ref, on=["gate", "inputs", "corner_id"], suffixes=("", "_spice"))
    assert len(g) == len(ref) == 3600
    # reltol-level agreement + an absolute floor: the source-referenced formula drops Is, so
    # BSIM4's ~1 fA per-device KCL residual (Part A1) times its node voltage remains (<= 6 devices)
    assert np.allclose(g.i_leak, g.i_leak_spice, rtol=1e-6, atol=2e-14)
    assert np.allclose(g.i_vdd, g.i_vdd_spice, rtol=2e-6, atol=1e-18)       # KCL at node vdd
    a = g.i_in_a_spice.notna()
    assert np.allclose(g.i_in_a[a], g.i_in_a_spice[a], rtol=1e-5, atol=1e-18)


@pytest.mark.skipif(not (HAVE_REFS and MODELS), reason="needs references and trained models")
def test_prediction_runs_no_spice(monkeypatch):
    import subprocess

    from leakage import simulator, spice, transistor
    from leakage.gate_ml import predict_gate_records

    def boom(*a, **k):
        raise AssertionError("SPICE must not run during gate inference")
    monkeypatch.setattr(subprocess, "run", boom)                  # any external process
    monkeypatch.setattr(simulator.NgspiceSimulator, "run_op", boom)
    monkeypatch.setattr(spice, "run_ngspice", boom)
    monkeypatch.setattr(transistor, "run_ngspice", boom)
    ref = pd.read_csv(GDIR / "gate_spice_results.csv", dtype={"inputs": str})
    sub = ref.groupby(["gate", "inputs"]).head(2)
    g, d = predict_gate_records(sub)
    assert len(g) == 36 and (g.i_leak > 0).all() and np.isfinite(g.i_leak).all()
    assert len(d) == sum(len(GATES[x].devices) for x in sub.gate)
