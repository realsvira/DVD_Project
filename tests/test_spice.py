import math

import pytest

from leakage.config import load_config
from leakage.spice import (find_ngspice, model_info, model_text_with, parse_prints,
                           read_model_nominals, run_ngspice, spice_path)

CFG = load_config()
NGSPICE = find_ngspice(CFG.ngspice_path)
needs_ngspice = pytest.mark.skipif(NGSPICE is None, reason="ngspice not found")

# values copied from models/45nm_HP.pm
EXPECTED = {
    "nmos": {"vth0": 0.46893, "toxe": 1.25e-9, "ndep": 3.24e18, "u0": 0.054, "xj": 1.4e-8},
    "pmos": {"vth0": -0.49158, "toxe": 1.3e-9, "ndep": 2.44e18, "u0": 0.02, "xj": 1.4e-8},
}


def test_model_cards_are_bsim4():
    info = model_info(CFG.spice_model_file)
    assert set(info) == {"nmos", "pmos"}
    for dev in ("nmos", "pmos"):
        assert info[dev]["type"] == dev
        assert info[dev]["level"] == 54


def test_nominals_parsed():
    nom = read_model_nominals(CFG.spice_model_file, CFG.process_params)
    for dev, vals in EXPECTED.items():
        for p, v in vals.items():
            assert math.isclose(nom[dev][p], v, rel_tol=1e-12), (dev, p)


def test_override_changes_only_target_model(tmp_path):
    txt = model_text_with(CFG.spice_model_file, {"nmos": {"toxe": 1.1e-9, "vth0": 0.5}})
    f = tmp_path / "m.pm"
    f.write_text(txt)
    nom = read_model_nominals(f, CFG.process_params)
    assert nom["nmos"]["toxe"] == 1.1e-9 and nom["nmos"]["vth0"] == 0.5
    assert nom["pmos"] == EXPECTED["pmos"]
    # toxm / toxref must remain untouched (only toxe is varied)
    assert model_info(f)["nmos"]["params"]["toxm"] == "1.25e-009"


def test_only_returns_single_card():
    txt = model_text_with(CFG.spice_model_file, only="pmos")
    assert txt.lower().count(".model") == 1 and "pmos" in txt.lower()


def test_parse_prints_tagged():
    out = "@@S 0\ni(vd) = -1.5e-09\ni(vg) = 2e-12\n@@S 1\ni(vd) = 3.0e-09\n"
    r = parse_prints(out)
    assert r[0]["i(vd)"] == -1.5e-9 and r[0]["i(vg)"] == 2e-12 and r[1]["i(vd)"] == 3e-9


RAW = """Title: * test
Date: today
Plotname: Operating Point
Flags: real
No. Variables: 3
No. Points: 2
Variables:
\t0\tv(d)\tvoltage
\t1\ti(vd)\tcurrent
\t2\tI(VG)\tcurrent
Values:
 0\t1.000000000000000e+00
\t-1.694674617662751e-09
\t1.28E-11
 1\t5.0e-01
\t-3.2e-10
\t-7e-15
"""


def test_parse_raw_ascii_handles_sign_exponent_case_and_points():
    from leakage.spice import parse_raw_ascii
    p = parse_raw_ascii(RAW)[0]
    assert p["variables"] == ["v(d)", "i(vd)", "i(vg)"]
    assert p["values"].shape == (2, 3)
    assert p["values"][0, 1] == -1.694674617662751e-09
    assert p["values"][0, 2] == 1.28e-11
    assert p["values"][1, 2] == -7e-15


def test_parse_raw_ascii_rejects_truncated():
    from leakage.spice import parse_raw_ascii
    with pytest.raises(ValueError):
        parse_raw_ascii(RAW.rsplit("\t-7e-15", 1)[0])


@needs_ngspice
def test_ngspice_runs_and_kcl_holds():
    net = (f"* kcl\n.include {spice_path(CFG.spice_model_file)}\n{CFG.spice_options}\n.temp 85\n"
           "M1 d g s b nmos W=90n L=45n\nVd d 0 1.0\nVg g 0 0\nVs s 0 0\nVb b 0 0\n"
           ".control\nset numdgt=10\nop\nprint i(vd) i(vg) i(vs) i(vb)\n.endc\n.end\n")
    v = parse_prints(run_ngspice(NGSPICE, net, timeout=120))[None]
    cur = [-v[f"i(v{t})"] for t in "dgsb"]
    assert cur[0] > 0                      # OFF drain leakage flows into the drain
    assert abs(sum(cur)) <= 1e-6 * max(map(abs, cur)) + 1e-15
