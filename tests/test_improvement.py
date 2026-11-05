import numpy as np
import pandas as pd
import pytest

from leakage.config import load_config
from leakage.ml import INPUTS, TARGETS, CurrentModel, EnsembleModel, TargetTransform
from leakage.transistor import nominal_process

CFG = load_config()
torch = pytest.importorskip("torch")


def toy(n=300, seed=0):
    from leakage.dataset import generate_samples
    df = generate_samples(CFG, "nmos_off", range(n))
    df["id"] = np.sinh(3 * (df.vd - df.vs)) * 10 ** (-9 + df.temp / 100)
    df["ig"] = -1e-11 * (1 + df.vd) * (df.w / 1e-7)
    df["ib"] = -1e-14 * (1 + df.vs)
    df["is"] = -(df["id"] + df["ig"] + df["ib"])
    return df


def model(seed=1, it=150):
    return CurrentModel("nmos_off", "nmos", "mlp_torch", {"hidden": (16, 16), "alpha": 1e-5, "max_iter": it,
                                                          "threads": 1},
                        nominal_process(CFG, "nmos"), TargetTransform("asinh", 1e-15), seed=seed, feature_set="v2")


def test_torch_mlp_reproducible_and_numpy_inference(tmp_path, monkeypatch):
    import joblib
    df = toy()
    a, b = model().fit(df), model().fit(df)
    pa = a.predict(df[INPUTS])
    pd.testing.assert_frame_equal(pa, b.predict(df[INPUTS]))          # same seed + threads -> identical
    assert a.estimator.n_iter_ > 0 and np.isfinite(a.estimator.loss_)
    joblib.dump(a, tmp_path / "m.joblib")
    import sys
    monkeypatch.setitem(sys.modules, "torch", None)                    # prediction must not need torch
    pd.testing.assert_frame_equal(pa, joblib.load(tmp_path / "m.joblib").predict(df[INPUTS]))


def test_torch_mlp_training_reduces_loss():
    df = toy()
    short, long_ = model(it=20).fit(df), model(it=300).fit(df)
    assert long_.estimator.loss_ < short.estimator.loss_


def test_ensemble_is_mean_in_asinh_space():
    df = toy()
    m1, m2 = model(seed=1).fit(df), model(seed=2).fit(df)
    e = EnsembleModel([m1, m2])
    tf = TargetTransform("asinh", 1e-15)
    expect = tf.inverse((tf.forward(m1.predict(df[INPUTS])[TARGETS].to_numpy())
                         + tf.forward(m2.predict(df[INPUTS])[TARGETS].to_numpy())) / 2)
    assert np.allclose(e.predict(df[INPUTS])[TARGETS].to_numpy(), expect, rtol=1e-12)
    assert e.case == "nmos_off" and e.device == "nmos" and e.algo == "ensemble"
    with pytest.raises(ValueError):
        EnsembleModel([])


def test_final_models_are_the_selected_v3():
    import hashlib
    for c in CFG.cases:
        f, v3 = CFG.trained_models / f"{c}.joblib", CFG.trained_models / "v3" / f"{c}.joblib"
        if not v3.exists():
            pytest.skip("v3 not created")
        assert hashlib.sha256(f.read_bytes()).digest() == hashlib.sha256(v3.read_bytes()).digest()
