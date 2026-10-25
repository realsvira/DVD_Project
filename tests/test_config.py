from leakage.config import load_config


def test_assignment_values():
    cfg = load_config()
    assert cfg.model_name == "45nm_HP"
    assert cfg.vdd_nominal == 1.0
    assert cfg.lmin == 45e-9
    assert cfg.test_circuit == "CLA3"


def test_paths_are_absolute_and_model_exists():
    cfg = load_config()
    assert cfg.spice_model_file.is_absolute()
    assert cfg.spice_model_file.is_file()
    assert cfg.spice_model_file.suffix == ".pm"


def test_sampling_ranges_follow_spec():
    cfg = load_config()
    lo, hi = cfg.vdd_range
    assert abs(lo - 0.9) < 1e-12 and abs(hi - 1.1) < 1e-12
    assert cfg.w_range == (45e-9, 8 * 45e-9)
    assert cfg.temp_range == (-40.0, 125.0)
    assert cfg.process_params == ("vth0", "toxe", "ndep", "u0", "xj")
    assert abs(sum(cfg.split.values()) - 1.0) < 1e-12


def test_cases_bias_table():
    cfg = load_config()
    c = cfg.cases
    assert (c["nmos_on"].vg_frac, c["nmos_on"].vb_frac) == (1.0, 0.0)
    assert (c["nmos_off"].vg_frac, c["nmos_off"].vb_frac) == (0.0, 0.0)
    assert (c["pmos_on"].vg_frac, c["pmos_on"].vb_frac) == (0.0, 1.0)
    assert (c["pmos_off"].vg_frac, c["pmos_off"].vb_frac) == (1.0, 1.0)


def test_dataset_columns():
    cfg = load_config()
    assert list(cfg.columns) == ["vdd", "temp", "w", "vd", "vs", "vg", "vb", "vth0", "toxe",
                                 "ndep", "u0", "xj", "id", "ig", "is", "ib"]
