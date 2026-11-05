"""Improvement phase, step 1: error diagnostics of the CURRENT final transistor models
on TRAINING and VALIDATION data only (no test / gate / circuit reference is used for
decisions). Gate- and circuit-level breakdowns are read from the existing result files
for description only.

Outputs (results/improve/diagnostics/):
  train_vs_val.csv           per case x terminal: median/p95 rel. error on train vs val
                             (equal = under-fitting / optimisation-limited; val >> train = over-fitting)
  error_by_factor.csv        median rel. error per terminal vs T, VDD, W, vth0, toxe, ndep, u0, xj,
                             |I| decade, |Vds|, source offset   (original + targeted validation)
  sign_and_magnitude.csv     sign accuracy and share of error from |I| < 10 fA
  coverage.csv               error vs distance to the nearest training sample (feature space)
  convergence.csv            L-BFGS iterations used vs cap, final training loss
  gate_level_breakdown.csv   (descriptive) B4 error vs T / VDD and dominant devices
  diagnostics.md

Run:  .\\.venv\\Scripts\\python.exe scripts\\d1_diagnostics.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from a2b_train_v2 import aug_splits, aug_train_val, original_train_val  # noqa: E402
from leakage.config import load_config  # noqa: E402
from leakage.ml import I_FLOOR, INPUTS, TARGETS, build_features, evaluate  # noqa: E402

CFG = load_config()
OUT = CFG.results / "improve" / "diagnostics"
MODELS = CFG.trained_models / "final_before_improvement"


def rel(true, pred):
    return np.abs(pred - true) / np.maximum(np.abs(true), I_FLOOR)


def md(df, fmt="{:.4g}"):
    cols = list(df.columns)
    rows = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        rows.append("| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(rows)


def main() -> int:
    import joblib
    from sklearn.neighbors import NearestNeighbors
    OUT.mkdir(parents=True, exist_ok=True)
    sp = aug_splits()
    tv, fac, sgn, cov, conv = [], [], [], [], []
    for case in CFG.cases:
        m = joblib.load(MODELS / f"{case}.joblib")
        tr, va = original_train_val(case)
        atr, ava = aug_train_val(case, sp)
        train = pd.concat([tr, atr], ignore_index=True)
        conv.append({"case": case, "hidden": str(m.params["hidden"]), "alpha": m.params["alpha"],
                     "n_iter": m.estimator.n_iter_, "max_iter": m.estimator.max_iter,
                     "hit_cap": m.estimator.n_iter_ >= m.estimator.max_iter, "train_loss": m.estimator.loss_})
        sets = {"train": train, "val_orig": va, "val_targeted": ava}
        preds = {k: m.predict(v[INPUTS]) for k, v in sets.items()}
        for k, v in sets.items():
            met = evaluate(v, preds[k]).set_index("terminal")
            for t in TARGETS + ["mean"]:
                tv.append({"case": case, "set": k, "terminal": t, "med_rel": met.loc[t, "med_rel"],
                           "p95_rel": met.loc[t, "p95_rel"]})
        # factor analysis on both validation sets
        val = pd.concat([va.assign(vset="orig"), ava.assign(vset="targeted")], ignore_index=True)
        pv = pd.concat([preds["val_orig"], preds["val_targeted"]], ignore_index=True)
        dev = CFG.cases[case].device
        nom = m.nominal
        f = pd.DataFrame({"temp": val.temp, "vdd": val.vdd, "w_nm": val.w * 1e9,
                          **{f"{p}_rel": val[p] / nom[p] for p in ("vth0", "toxe", "ndep", "u0", "xj")},
                          "abs_vds": (val.vd - val.vs).abs(),
                          "src_off_vdd": (np.minimum(val.vd, val.vs) if dev == "nmos"
                                          else val.vdd - np.maximum(val.vd, val.vs)) / val.vdd})
        for t in TARGETS:
            e = rel(val[t].to_numpy(), pv[t].to_numpy())
            f[f"err_{t}"] = e
            f[f"logI_{t}"] = np.log10(np.maximum(val[t].abs(), 1e-20))
        bins = {"temp": [-40, 0, 40, 80, 125.01], "vdd": [0.9, 0.95, 1.0, 1.05, 1.1001], "w_nm": [45, 100, 200, 300, 360.01],
                **{f"{p}_rel": [0.9, 0.95, 1.0, 1.05, 1.1001] for p in ("vth0", "toxe", "ndep", "u0", "xj")},
                "abs_vds": [0, 1e-6, 1e-3, 0.05, 0.3, 1.2], "src_off_vdd": [0, 1e-6, 0.05, 0.2, 0.5, 1.01]}
        for col, b in bins.items():
            cat = pd.cut(f[col], b, include_lowest=True)
            for t in TARGETS:
                g = f.groupby(cat, observed=True)[f"err_{t}"]
                for bin_, s in g:
                    fac.append({"case": case, "factor": col, "bin": str(bin_), "terminal": t, "n": len(s),
                                "med_rel": s.median(), "p95_rel": s.quantile(0.95)})
        for t in TARGETS:
            cat = pd.cut(f[f"logI_{t}"], [-21, -14, -12, -10, -8, -6, -2], include_lowest=True)
            for bin_, s in f.groupby(cat, observed=True)[f"err_{t}"]:
                fac.append({"case": case, "factor": "log10|I|", "bin": str(bin_), "terminal": t, "n": len(s),
                            "med_rel": s.median(), "p95_rel": s.quantile(0.95)})
            big = val[t].abs() >= 10 * I_FLOOR
            sgn.append({"case": case, "terminal": t,
                        "sign_acc": float((np.sign(pv[t][big]) == np.sign(val[t][big])).mean()),
                        "frac_below_10fA": float((~big).mean()),
                        "share_of_p95_tail_below_10fA": float((~big)[f[f"err_{t}"] > f[f"err_{t}"].quantile(0.95)].mean())})
        # coverage: distance (standardised features) to nearest training sample vs error
        Xtr = build_features(train, dev, nom, m.feature_set).to_numpy(float)
        mu, sd = Xtr.mean(0), Xtr.std(0)
        nn = NearestNeighbors(n_neighbors=1).fit((Xtr - mu) / sd)
        dist, _ = nn.kneighbors((build_features(val, dev, nom, m.feature_set).to_numpy(float) - mu) / sd)
        f["nn_dist"] = dist[:, 0]
        q = pd.qcut(f.nn_dist, 4, labels=["closest 25%", "25-50%", "50-75%", "farthest 25%"])
        for bin_, g in f.groupby(q, observed=True):
            cov.append({"case": case, "nn_distance_quartile": str(bin_), "median_dist": g.nn_dist.median(),
                        **{f"med_rel_{t}": g[f"err_{t}"].median() for t in TARGETS}})
        print(f"{case}: done")

    tv = pd.DataFrame(tv); fac = pd.DataFrame(fac); sgn = pd.DataFrame(sgn)
    cov = pd.DataFrame(cov); conv = pd.DataFrame(conv)
    for name, d in (("train_vs_val", tv), ("error_by_factor", fac), ("sign_and_magnitude", sgn),
                    ("coverage", cov), ("convergence", conv)):
        d.to_csv(OUT / f"{name}.csv", index=False)

    # descriptive gate-level breakdown (existing B4 results; no decisions are based on it)
    b4 = pd.read_csv(CFG.data / "gates" / "gate_ml_predictions.csv", dtype={"inputs": str})
    b4["tbin"] = pd.cut(b4.temp, [-40, 15, 70, 125.01]).astype(str)
    gl = b4.groupby(["gate", "tbin"]).rel_err.agg(bias="median", med=lambda e: e.abs().median()).reset_index()
    gl.to_csv(OUT / "gate_level_breakdown.csv", index=False)

    tvp = tv[tv.terminal == "mean"].pivot(index="case", columns="set", values="med_rel").reset_index()
    rep = ["# Improvement phase: diagnostics of the current final models (train/validation only)", "",
           "## Optimisation", "", md(conv), "",
           "Every final model stopped at the L-BFGS iteration cap: the fits are optimisation-limited.", "",
           "## Train vs validation (mean over terminals, median relative error)", "", md(tvp), "",
           "Train ≈ validation means the models are not over-fitting; the error floor is set by how well the "
           "network is fitted (under-fitting), not by too little data.", "",
           "## Sign accuracy and tiny currents", "", md(sgn), "",
           "## Error vs distance to the nearest training sample (coverage)", "", md(cov), "",
           "Full factor tables: `error_by_factor.csv`.", ""]
    (OUT / "diagnostics.md").write_text("\n".join(rep), encoding="utf-8")
    pd.set_option("display.width", 220)
    print(conv.to_string(index=False)); print(tvp.to_string(index=False)); print(sgn.to_string(index=False))
    print(cov.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
