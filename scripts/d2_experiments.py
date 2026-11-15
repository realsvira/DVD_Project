"""Improvement phase, step 2: candidate models scored on VALIDATION ONLY with the
pre-declared score (results/improve/selection_rule.md).

Experiments (each result cached in results/improve/experiments/<exp>/<case>.joblib + .json):
  E0_baseline        current final models (v2, sklearn L-BFGS, stopped at the 3,000-iteration cap)
  E1_ens_v2cache     mean (asinh space) of the 3 already-trained v2 MLPs of the case (no new training)
  E2_torch_same      same architecture/alpha as the baseline, PyTorch L-BFGS, larger iteration budget
  E3_torch_wide      (256,256,256), PyTorch L-BFGS
  E4_ens_torch       mean of E2 with 3 seeds
  LC_<frac>          learning curve: E2 settings on a random fraction of the training data

Run:  .venv/bin/python scripts/d2_experiments.py --exp E1_ens_v2cache E2_torch_same ...
      .venv/bin/python scripts/d2_experiments.py --summary
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from a2b_train_v2 import aug_splits, aug_train_val, original_train_val  # noqa: E402
from leakage.config import load_config  # noqa: E402
from leakage.ml import INPUTS, TARGETS, CurrentModel, EnsembleModel, TargetTransform, evaluate  # noqa: E402
from leakage.transistor import nominal_process  # noqa: E402

CFG = load_config()
OUT = CFG.results / "improve" / "experiments"
BASE = CFG.trained_models / "final_before_improvement"
CACHE_V2 = CFG.results / "a2b" / "model_cache"
SEED = CFG.seeds["training"]


def score(model, sets) -> dict:
    """Pre-declared score: mean over val sets of [mean median + 0.1 * mean p95] (over 4 terminals)."""
    out, s = {}, []
    for name, df in sets.items():
        met = evaluate(df, model.predict(df[INPUTS])).set_index("terminal")
        med, p95 = met.loc["mean", "med_rel"], met.loc["mean", "p95_rel"]
        out[f"{name}_med"], out[f"{name}_p95"] = float(med), float(p95)
        for t in TARGETS:
            out[f"{name}_{t}_med"] = float(met.loc[t, "med_rel"])
        s.append(med + 0.1 * p95)
    out["score"] = float(np.mean(s))
    return out


def data(case, sp):
    tr, va = original_train_val(case)
    atr, ava = aug_train_val(case, sp)
    return pd.concat([tr, atr], ignore_index=True), {"val_orig": va, "val_targeted": ava}


def base_params(case):
    import joblib
    return joblib.load(BASE / f"{case}.joblib").params


def fit_torch(case, train, hidden, alpha, max_iter, seed, threads):
    dev = CFG.cases[case].device
    m = CurrentModel(case, dev, "mlp_torch", {"hidden": tuple(hidden), "alpha": alpha, "max_iter": max_iter,
                                               "threads": threads},
                     nominal_process(CFG, dev), TargetTransform("asinh", 1e-15), seed=seed, feature_set="v2")
    return m.fit(train)


def build(exp, case, sp, max_iter, threads):
    import joblib
    train, _ = data(case, sp)
    bp = base_params(case)
    if exp == "E0_baseline":
        return joblib.load(BASE / f"{case}.joblib")
    if exp == "E1_ens_v2cache":
        return EnsembleModel([joblib.load(CACHE_V2 / f"{case}__{i}.joblib") for i in range(3)],
                             meta={"experiment": exp})
    if exp == "E2_torch_same":
        return fit_torch(case, train, bp["hidden"], bp["alpha"], max_iter, SEED, threads)
    if exp == "E3_torch_wide":
        return fit_torch(case, train, (256, 256, 256), bp["alpha"], max_iter, SEED, threads)
    if exp == "E4_ens_torch":
        prev = OUT / "E2_torch_same" / f"{case}.joblib"
        members = [joblib.load(prev)] if prev.exists() else [fit_torch(case, train, bp["hidden"], bp["alpha"],
                                                                         max_iter, SEED, threads)]
        members += [fit_torch(case, train, bp["hidden"], bp["alpha"], max_iter, SEED + k, threads) for k in (1, 2)]
        return EnsembleModel(members, meta={"experiment": exp})
    if exp == "E5_torch_moredata":
        # + all of the second targeted batch (salt 6); validation sets unchanged
        aug2 = "transistor_aug2_on" if case.endswith("_on") else "transistor_aug2_off"
        extra = pd.read_csv(CFG.data / aug2 / f"{case}.csv")
        return fit_torch(case, pd.concat([train, extra], ignore_index=True), bp["hidden"], bp["alpha"],
                         max_iter, SEED, threads)
    if exp == "E6_ens_E2_E5":
        return EnsembleModel([joblib.load(OUT / e / f"{case}.joblib") for e in ("E2_torch_same", "E5_torch_moredata")],
                             meta={"experiment": exp})
    if exp.startswith("LC_"):
        frac = float(exp[3:])
        sub = train.sample(frac=frac, random_state=SEED).reset_index(drop=True)
        return fit_torch(case, sub, bp["hidden"], bp["alpha"], max_iter, SEED, threads)
    raise ValueError(exp)


def main() -> int:
    import joblib
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="*", default=[])
    ap.add_argument("--cases", nargs="+", default=list(CFG.cases))
    ap.add_argument("--max-iter", type=int, default=20000)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--select", action="store_true", help="pick the best experiment per case -> trained_models/v3")
    a = ap.parse_args()
    sp = aug_splits()
    for exp in a.exp:
        for case in a.cases:
            d = OUT / exp
            d.mkdir(parents=True, exist_ok=True)
            if (d / f"{case}.json").exists():
                print(f"{exp} {case}: cached")
                continue
            t0 = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = build(exp, case, sp, a.max_iter, a.threads)
            secs = time.perf_counter() - t0
            _, sets = data(case, sp)
            r = {"experiment": exp, "case": case, "seconds": round(secs, 1), **score(m, sets)}
            est = getattr(m, "estimator", None)
            if est is not None:
                r.update({"n_iter": getattr(est, "n_iter_", None), "train_loss": getattr(est, "loss_", None)})
            if exp != "E0_baseline":
                joblib.dump(m, d / f"{case}.joblib", compress=3)
            (d / f"{case}.json").write_text(json.dumps(r, indent=1, default=str), encoding="utf-8")
            print(f"{exp:16s} {case:8s} score {r['score']:.4f}  orig {r['val_orig_med']:.4f}/{r['val_orig_p95']:.3f}  "
                  f"targeted {r['val_targeted_med']:.4f}/{r['val_targeted_p95']:.3f}  {secs:.0f}s "
                  f"iters={r.get('n_iter')}", flush=True)
    if a.select:
        rows = [json.loads(f.read_text(encoding="utf-8")) for f in OUT.glob("*/*.json")]
        t = pd.DataFrame(rows)
        v3 = CFG.trained_models / "v3"
        v3.mkdir(parents=True, exist_ok=True)
        sel = {}
        for case in CFG.cases:
            c = t[t.case == case].sort_values("score")
            best = c.iloc[0]
            src = (BASE / f"{case}.joblib") if best.experiment == "E0_baseline" else OUT / best.experiment / f"{case}.joblib"
            m = joblib.load(src)
            m.meta = {**(m.meta or {}), "selected_experiment": best.experiment, "validation_score": float(best.score),
                      "selection_rule": "results/improve/selection_rule.md",
                      "candidates": {r.experiment: float(r.score) for r in c.itertuples()}}
            joblib.dump(m, v3 / f"{case}.joblib", compress=3)
            sel[case] = {"selected": best.experiment, "score": float(best.score),
                         "baseline_score": float(c[c.experiment == "E0_baseline"].score.iloc[0]),
                         "candidates": {r.experiment: float(r.score) for r in c.itertuples()}}
            print(f"{case}: {best.experiment} score {best.score:.4f} (baseline {sel[case]['baseline_score']:.4f})")
        (CFG.results / "improve" / "selection.json").write_text(json.dumps(sel, indent=2), encoding="utf-8")
    if a.summary or not a.exp:
        rows = [json.loads(f.read_text(encoding="utf-8")) for f in OUT.glob("*/*.json")]
        t = pd.DataFrame(rows)
        t.to_csv(CFG.results / "improve" / "experiments_validation.csv", index=False)
        pd.set_option("display.width", 250)
        cols = ["experiment", "case", "score", "val_orig_med", "val_orig_p95", "val_targeted_med", "val_targeted_p95",
                "seconds", "n_iter"]
        print(t[[c for c in cols if c in t]].sort_values(["case", "score"]).to_string(index=False,
                                                                                       float_format=lambda x: f"{x:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
