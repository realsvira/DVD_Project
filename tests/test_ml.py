import numpy as np
import pandas as pd
import pytest

from leakage.config import load_config
from leakage.ml import (FEATURES, I_FLOOR, INPUTS, TARGETS, CurrentModel, TargetTransform,
                        build_features, evaluate, terminal_metrics)
from leakage.transistor import nominal_process

CFG = load_config()
NOM = {d: nominal_process(CFG, d) for d in ("nmos", "pmos")}
MODELS_EXIST = all((CFG.trained_models / f"{c}.joblib").is_file() for c in CFG.cases)


def frame(n=200, seed=0, case="nmos_off"):
    from leakage.dataset import generate_samples
    df = generate_samples(CFG, case, range(n))
    rng = np.random.default_rng(seed)
    # synthetic smooth signed currents spanning decades (KCL-consistent)
    df["id"] = np.sinh(3 * (df.vd - df.vs)) * 10 ** (-9 + df.temp / 100)
    df["ig"] = -1e-11 * (1 + df.vd) * (df.w / 1e-7)
    df["ib"] = -1e-14 * (1 + 0.1 * rng.standard_normal(n))
    df["is"] = -(df["id"] + df["ig"] + df["ib"])
    return df


def test_transform_roundtrip_preserves_sign_and_zero():
    x = np.array([-1e-3, -1e-12, -1e-16, 0.0, 1e-16, 1e-12, 1e-3])
    for tf in (TargetTransform("asinh", 1e-15), TargetTransform("asinh", 1e-12), TargetTransform("linear")):
        y = tf.inverse(tf.forward(x))
        assert np.allclose(y, x, rtol=1e-12, atol=0)
        assert (np.sign(y) == np.sign(x)).all()


def test_features_nmos_and_pmos():
    df = pd.DataFrame({"vdd": [1.0], "temp": [27.0], "w": [1e-7], "vd": [0.8], "vs": [0.3], "vg": [0.0],
                       "vb": [0.0], **{p: [NOM["nmos"][p]] for p in NOM["nmos"]}})
    f = build_features(df, "nmos", NOM["nmos"]).iloc[0]
    assert list(f.index) == FEATURES
    assert f.vds == pytest.approx(0.5) and f.vgs == pytest.approx(-0.3) and f.vgd == pytest.approx(-0.8)
    assert f.src_off == pytest.approx(0.3) and f.temp_k == pytest.approx(300.15)
    assert f.logw == pytest.approx(-7.0) and f.vth0_rel == pytest.approx(1.0)
    dfp = df.assign(vg=1.0, vb=1.0, **{p: NOM["pmos"][p] for p in NOM["pmos"]})
    fp = build_features(dfp, "pmos", NOM["pmos"]).iloc[0]
    assert fp.src_off == pytest.approx(0.2)        # VDD - max(vd, vs)


def test_feature_set_v2_adds_signed_log_vds():
    from leakage.ml import FEATURE_SETS
    df = pd.DataFrame({"vdd": [1.0, 1.0], "temp": [27.0, 27.0], "w": [1e-7, 1e-7], "vd": [0.5, 0.5],
                       "vs": [0.5 - 1e-6, 0.5 + 1e-3], "vg": [1.0, 1.0], "vb": [0.0, 0.0],
                       **{p: [NOM["nmos"][p]] * 2 for p in NOM["nmos"]}})
    f = build_features(df, "nmos", NOM["nmos"], "v2")
    assert list(f.columns) == FEATURE_SETS["v2"] and FEATURE_SETS["v2"][:-1] == FEATURES
    assert f.vds_slog.iloc[0] == pytest.approx(np.arcsinh(1.0), rel=1e-6)
    assert f.vds_slog.iloc[1] < -7                                            # ~ -ln(2000)


def test_metric_definitions():
    true = np.array([1e-9, -2e-12, 1e-17, 5e-6])
    pred = np.array([1.01e-9, -2.2e-12, 3e-17, -5e-6])
    scale = np.array([1e-9, 2e-12, 1e-15, 5e-6])
    m = terminal_metrics(true, pred, scale)
    rel = np.array([0.01, 0.1, 2e-17 / I_FLOOR, 2.0])
    assert m["med_rel"] == pytest.approx(np.median(rel))
    assert m["sign_acc"] == pytest.approx(2 / 3)        # 1e-17 excluded (< 10 fA)
    assert m["med_abs"] == pytest.approx(np.median(np.abs(pred - true)))


def test_evaluate_has_mean_row():
    df = frame(50)
    out = evaluate(df, df[TARGETS] * 1.02)
    assert list(out.terminal) == TARGETS + ["mean"]
    assert np.allclose(out.med_rel[:4], 0.02)


@pytest.mark.parametrize("algo,params", [("poly_ridge", {"degree": 2, "alpha": 1e-6}),
                                         ("mlp", {"hidden": (16,), "alpha": 1e-5, "max_iter": 200})])
def test_model_fit_predict_and_joblib_roundtrip(tmp_path, algo, params):
    import joblib
    df = frame(300)
    m = CurrentModel("nmos_off", "nmos", algo, params, NOM["nmos"], TargetTransform("asinh", 1e-15),
                     seed=1).fit(df)
    p = m.predict(df[INPUTS])
    assert list(p.columns) == TARGETS and np.isfinite(p.to_numpy()).all()
    joblib.dump(m, tmp_path / "m.joblib")
    m2 = joblib.load(tmp_path / "m.joblib")
    pd.testing.assert_frame_equal(p, m2.predict(df[INPUTS]))


def test_split_file_disjoint_and_sized():
    f = CFG.data / "transistor" / "splits.csv"
    if not f.exists():
        pytest.skip("splits not created yet")
    s = pd.read_csv(f)
    for case, g in s.groupby("case"):
        assert g.sample_id.is_unique
        assert g.split.value_counts().to_dict() == {"train": 7000, "val": 1500, "test": 1500}


@pytest.mark.skipif(not MODELS_EXIST, reason="trained models not present")
def test_inference_api_against_spice():
    from leakage.inference import LeakagePredictor
    from leakage.transistor import simulate_transistor
    lp = LeakagePredictor()
    for dev, kw in (("nmos", dict(vd=1.0, vg=0.0, vs=0.0, vb=0.0)), ("pmos", dict(vd=0.0, vg=1.0, vs=1.0, vb=1.0))):
        r = lp.predict(dev, vdd=1.0, temp=27.0, w=90e-9, **kw).iloc[0]
        assert r.case == f"{dev}_off" and not r.extrapolated
        sp = simulate_transistor(CFG, f"{dev}_off", temp=27.0, w=90e-9)
        assert abs(r["id"] - sp["id"]) / abs(sp["id"]) < 0.5          # loose sanity bound
        assert np.sign(r["id"]) == np.sign(sp["id"])


@pytest.mark.skipif(not MODELS_EXIST, reason="trained models not present")
def test_inference_rules_and_vectorisation():
    from leakage.inference import LeakagePredictor
    lp = LeakagePredictor()
    out = lp.predict("nmos", vd=[1.0, 0.0], vg=[0.0, 1.0], vs=0.0, vb=0.0, vdd=1.0, temp=27.0, w=90e-9)
    assert list(out.case) == ["nmos_off", "nmos_on"]
    with pytest.raises(ValueError):
        lp.predict("nmos", vd=1.0, vg=0.5, vs=0.0, vb=0.0, vdd=1.0, temp=27.0, w=90e-9)   # Vg not 0/VDD
    with pytest.raises(ValueError):
        lp.predict("nmos", vd=1.0, vg=0.0, vs=0.0, vb=0.3, vdd=1.0, temp=27.0, w=90e-9)   # Vb rule
    ext = lp.predict("nmos", vd=1.0, vg=0.0, vs=0.0, vb=0.0, vdd=1.0, temp=150.0, w=90e-9)
    assert bool(ext.extrapolated.iloc[0])
