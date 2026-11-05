"""Part A2 machine-learning core: features, target transforms, error metrics and
the CurrentModel wrapper that is trained, saved (joblib) and used for inference.

Conventions
-----------
* Targets: the four terminal currents id, ig, is, ib in A, positive INTO the
  device (same as the datasets). Signs are preserved end-to-end.
* One model per transistor case (nmos_on, nmos_off, pmos_on, pmos_off); each
  predicts all four currents (multi-output regression).

Target transform
----------------
Currents span ~11 decades and change sign. The default transform is the
sign-preserving inverse hyperbolic sine
        y = asinh(I / I0),   I = I0 * sinh(y)
which behaves like sign(I)*ln(2|I|/I0) for |I| >> I0 and is linear and smooth
through zero, so zero crossings (e.g. OFF-state Id) are not a discontinuity.
`linear` (y = I, standardised) is available for comparison. Targets are then
standardised per terminal with statistics from the TRAINING split only.

Error metrics (per terminal, all computed on currents in A)
-------------------------------------------------------------
  rel_err   = |I_pred - I_spice| / max(|I_spice|, I_FLOOR),   I_FLOOR = 1e-15 A (1 fA)
              The floor only matters for |I| < 1 fA (a handful of samples), where a
              relative error is not physically meaningful.
  abs_err   = |I_pred - I_spice|  (A)
  scale_err = |I_pred - I_spice| / max_k |I_spice,k|   (error relative to the largest
              terminal current of that sample = significance for total leakage)
  sign_acc  = fraction of samples with sign(I_pred) == sign(I_spice), counted only
              where |I_spice| >= 10 * I_FLOOR (sign of a sub-fA current is meaningless)
  R2_t      = coefficient of determination of asinh(I / 1 fA)  (common space for
              every model, independent of the training transform)
  R2_raw    = R^2 on the currents in A (dominated by the largest currents)
Reported as median and 95th percentile. "Accuracy" where quoted = 100 % - rel_err.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TARGETS = ["id", "ig", "is", "ib"]
INPUTS = ["vdd", "temp", "w", "vd", "vs", "vg", "vb", "vth0", "toxe", "ndep", "u0", "xj"]
PROCESS = ["vth0", "toxe", "ndep", "u0", "xj"]
I_FLOOR = 1e-15
R2_I0 = 1e-15

FEATURES = ["vdd", "temp_k", "logw", "vd", "vs", "vds", "avds", "vgs", "vgd", "vbs", "vbd",
            "src_off", "vth0_rel", "toxe_rel", "ndep_rel", "u0_rel", "xj_rel"]
# v2 adds a signed-log of Vds: near Vds = 0 the channel current is ~G*Vds, so
# asinh(Id/I0) ~ asinh(Vds/V0) + const -- a feature the network can use linearly.
VDS_SLOG_V0 = 1e-6
FEATURE_SETS = {"v1": FEATURES, "v2": FEATURES + ["vds_slog"]}


# ===================================================================== features
def build_features(df: pd.DataFrame, device: str, nominal: dict[str, float],
                   feature_set: str = "v1") -> pd.DataFrame:
    """Physics-motivated features. Identical code is used in training and inference.

    src_off: distance of the effective source from its rail (NMOS: min(Vd, Vs);
    PMOS: VDD - max(Vd, Vs)); the data check showed OFF current falls ~1 decade
    per 0.1 VDD of it. Process parameters are given relative to the device's
    nominal value (1.0 = nominal) so they are O(1)."""
    vd, vs, vg, vb, vdd = (df[c].to_numpy(float) for c in ("vd", "vs", "vg", "vb", "vdd"))
    if device == "nmos":
        src_off = np.minimum(vd, vs)
    elif device == "pmos":
        src_off = vdd - np.maximum(vd, vs)
    else:
        raise ValueError(f"unknown device {device}")
    f = pd.DataFrame({
        "vdd": vdd,
        "temp_k": df["temp"].to_numpy(float) + 273.15,
        "logw": np.log10(df["w"].to_numpy(float)),
        "vd": vd, "vs": vs,
        "vds": vd - vs, "avds": np.abs(vd - vs),
        "vgs": vg - vs, "vgd": vg - vd, "vbs": vb - vs, "vbd": vb - vd,
        "src_off": src_off,
        "vds_slog": np.arcsinh((vd - vs) / VDS_SLOG_V0),
    }, index=df.index)
    for p in PROCESS:
        f[f"{p}_rel"] = df[p].to_numpy(float) / nominal[p]
    return f[FEATURE_SETS[feature_set]]


# ===================================================================== transforms
@dataclass
class TargetTransform:
    kind: str = "asinh"      # "asinh" | "linear"
    i0: float = 1e-15        # asinh scale (A)

    def forward(self, current: np.ndarray) -> np.ndarray:
        current = np.asarray(current, float)
        if self.kind == "asinh":
            return np.arcsinh(current / self.i0)
        if self.kind == "linear":
            return current.copy()
        raise ValueError(self.kind)

    def inverse(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, float)
        if self.kind == "asinh":
            return self.i0 * np.sinh(np.clip(y, -700, 700))
        if self.kind == "linear":
            return y.copy()
        raise ValueError(self.kind)

    @property
    def name(self) -> str:
        return f"asinh(I/{self.i0:g})" if self.kind == "asinh" else "linear"


# ===================================================================== estimators
ALGORITHMS = ["poly_ridge", "random_forest", "lightgbm", "mlp"]


def make_estimator(algo: str, params: dict, seed: int, n_jobs: int = 1):
    """Multi-output regressor for standardised features -> standardised targets."""
    if algo == "poly_ridge":
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import PolynomialFeatures, StandardScaler
        return make_pipeline(PolynomialFeatures(degree=params["degree"], include_bias=False),
                             StandardScaler(), Ridge(alpha=params["alpha"]))
    if algo == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=params.get("n_estimators", 300),
                                     max_features=params["max_features"],
                                     min_samples_leaf=params["min_samples_leaf"],
                                     random_state=seed, n_jobs=n_jobs)
    if algo == "lightgbm":
        from lightgbm import LGBMRegressor
        from sklearn.multioutput import MultiOutputRegressor
        return MultiOutputRegressor(LGBMRegressor(
            n_estimators=params["n_estimators"], learning_rate=params["learning_rate"],
            num_leaves=params["num_leaves"], min_child_samples=params.get("min_child_samples", 10),
            subsample=0.8, subsample_freq=1, colsample_bytree=0.9,
            reg_lambda=params.get("reg_lambda", 0.0), random_state=seed, n_jobs=n_jobs, verbose=-1))
    if algo == "mlp":
        from sklearn.neural_network import MLPRegressor
        return MLPRegressor(hidden_layer_sizes=tuple(params["hidden"]), activation=params.get("activation", "tanh"),
                            alpha=params["alpha"], solver=params.get("solver", "lbfgs"),
                            max_iter=params.get("max_iter", 3000), tol=1e-7, random_state=seed,
                            max_fun=params.get("max_iter", 3000) * 2)
    if algo == "mlp_torch":
        from .torch_mlp import TorchMLPRegressor
        return TorchMLPRegressor(hidden=params["hidden"], alpha=params["alpha"],
                                 max_iter=params.get("max_iter", 20000), seed=seed,
                                 threads=params.get("threads", 4))
    if algo == "ensemble":
        # params["members"]: list of fitted CurrentModel objects; built by EnsembleModel below
        raise ValueError("ensembles are built with EnsembleModel, not make_estimator")
    raise ValueError(f"unknown algorithm {algo}")


def params_key(algo: str, params: dict) -> str:
    s = json.dumps({"algo": algo, **params}, sort_keys=True)
    return hashlib.sha1(s.encode()).hexdigest()[:10]


# ===================================================================== model wrapper
@dataclass
class CurrentModel:
    """Features -> four terminal currents for ONE transistor case."""
    case: str
    device: str
    algo: str
    params: dict
    nominal: dict
    transform: TargetTransform = field(default_factory=TargetTransform)
    seed: int = 0
    feature_set: str = "v1"      # class-level default keeps v1 pickles loadable
    feature_names: list = field(default_factory=lambda: list(FEATURES))
    target_names: list = field(default_factory=lambda: list(TARGETS))
    x_mean: np.ndarray | None = None
    x_std: np.ndarray | None = None
    y_mean: np.ndarray | None = None
    y_std: np.ndarray | None = None
    estimator: object = None
    meta: dict = field(default_factory=dict)

    def _x(self, df: pd.DataFrame) -> np.ndarray:
        X = build_features(df, self.device, self.nominal, self.feature_set).to_numpy(float)
        return (X - self.x_mean) / self.x_std

    def fit(self, df: pd.DataFrame, n_jobs: int = 1) -> "CurrentModel":
        self.feature_names = list(FEATURE_SETS[self.feature_set])
        X = build_features(df, self.device, self.nominal, self.feature_set).to_numpy(float)
        self.x_mean, self.x_std = X.mean(axis=0), X.std(axis=0)
        self.x_std[self.x_std == 0] = 1.0
        Y = self.transform.forward(df[TARGETS].to_numpy(float))
        self.y_mean, self.y_std = Y.mean(axis=0), Y.std(axis=0)
        self.y_std[self.y_std == 0] = 1.0
        self.estimator = make_estimator(self.algo, self.params, self.seed, n_jobs)
        self.estimator.fit((X - self.x_mean) / self.x_std, (Y - self.y_mean) / self.y_std)
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        Ys = np.asarray(self.estimator.predict(self._x(df)), float).reshape(len(df), -1)
        current = self.transform.inverse(Ys * self.y_std + self.y_mean)
        return pd.DataFrame(current, columns=TARGETS, index=df.index)


# ===================================================================== metrics
def _r2(a: np.ndarray, b: np.ndarray) -> float:
    ss = np.sum((a - a.mean()) ** 2)
    return float(1 - np.sum((a - b) ** 2) / ss) if ss > 0 else float("nan")


def terminal_metrics(true: np.ndarray, pred: np.ndarray, scale: np.ndarray) -> dict:
    err = np.abs(pred - true)
    rel = err / np.maximum(np.abs(true), I_FLOOR)
    sc = err / np.maximum(scale, I_FLOOR)
    signed = np.abs(true) >= 10 * I_FLOOR
    t = np.arcsinh(true / R2_I0)
    p = np.arcsinh(pred / R2_I0)
    return {
        "r2_t": _r2(t, p), "r2_raw": _r2(true, pred),
        "med_rel": float(np.median(rel)), "p95_rel": float(np.quantile(rel, 0.95)),
        "mean_rel": float(np.mean(rel)),
        "frac_rel_lt_1pct": float(np.mean(rel < 0.01)), "frac_rel_lt_5pct": float(np.mean(rel < 0.05)),
        "med_abs": float(np.median(err)), "p95_abs": float(np.quantile(err, 0.95)),
        "med_scale": float(np.median(sc)), "p95_scale": float(np.quantile(sc, 0.95)),
        "sign_acc": float(np.mean(np.sign(pred[signed]) == np.sign(true[signed]))) if signed.any() else float("nan"),
    }


def evaluate(true_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    """One row per terminal plus a 'mean' row (mean over the four terminals)."""
    T = true_df[TARGETS].to_numpy(float)
    P = pred_df[TARGETS].to_numpy(float)
    scale = np.abs(T).max(axis=1)
    rows = [{"terminal": t, **terminal_metrics(T[:, j], P[:, j], scale)} for j, t in enumerate(TARGETS)]
    m = pd.DataFrame(rows)
    mean = m.drop(columns="terminal").mean(numeric_only=True)
    return pd.concat([m, pd.DataFrame([{"terminal": "mean", **mean.to_dict()}])], ignore_index=True)


def selection_score(metrics: pd.DataFrame) -> float:
    """Validation selection criterion: mean over the four terminals of the median
    relative error (lower is better). Ties broken by the mean p95 relative error."""
    m = metrics[metrics.terminal != "mean"]
    return float(m.med_rel.mean())


# ===================================================================== ensembles
class EnsembleModel:
    """Average of several CurrentModels of the same case, taken in the target-transform
    (asinh) space and transformed back. Same interface as CurrentModel (predict, case,
    device, nominal, algo, params, meta), so LeakagePredictor / gate models use it as-is."""

    def __init__(self, members: list, meta: dict | None = None):
        if not members:
            raise ValueError("empty ensemble")
        cases = {m.case for m in members}
        if len(cases) != 1:
            raise ValueError(f"members from different cases: {cases}")
        self.members = list(members)
        m0 = members[0]
        self.case, self.device, self.nominal = m0.case, m0.device, m0.nominal
        self.transform = TargetTransform("asinh", 1e-15)
        self.algo = "ensemble"
        self.params = {"members": [{"algo": m.algo, **{k: (list(v) if isinstance(v, tuple) else v)
                                                         for k, v in m.params.items()},
                                    "feature_set": getattr(m, "feature_set", "v1")} for m in members]}
        self.feature_set = "+".join(sorted({getattr(m, "feature_set", "v1") for m in members}))
        self.meta = meta or {}

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        ys = [self.transform.forward(m.predict(df)[TARGETS].to_numpy(float)) for m in self.members]
        current = self.transform.inverse(np.mean(ys, axis=0))
        return pd.DataFrame(current, columns=TARGETS, index=df.index)
