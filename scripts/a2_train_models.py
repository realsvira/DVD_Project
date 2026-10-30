"""Part A2: train, tune and select one ML model per transistor case.

Phases (all use ONLY the training and validation splits; test rows are never loaded
into any model, scaler, transform choice or selection step):
  1. split     70/15/15 per case with a fixed seed; saved to data/transistor/splits.csv
               (created once, then re-used and verified).
  2. transform study: LightGBM (fixed settings) trained with several target
               transforms; the best by validation selects the transform for all models.
  3. tuning    grid search of 4 algorithm types x 4 cases, trained on train, scored on
               validation. Each (case, algorithm, config) result is cached in
               results/a2/tuning/ so an interrupted run resumes.
  4. selection best algorithm per case = lowest validation selection score
               (mean over Id/Ig/Is/Ib of the median relative error).
  5. save      the selected model, refit on the training split with the chosen
               config (deterministic), to trained_models/<case>.joblib with metadata.

Test-set evaluation is a separate, one-time step: scripts/a2_evaluate_test.py.

Run:
  .\\.venv\\Scripts\\python.exe scripts\\a2_train_models.py            (full)
  .\\.venv\\Scripts\\python.exe scripts\\a2_train_models.py --quick    (tiny grid, fast code check;
                                                                       writes to results/a2_quick, no models saved)
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from leakage.config import load_config  # noqa: E402
from leakage.ml import (ALGORITHMS, FEATURES, TARGETS, CurrentModel, TargetTransform,  # noqa: E402
                        evaluate, params_key, selection_score)
from leakage.transistor import nominal_process  # noqa: E402

CFG = load_config()
DATA = CFG.data / "transistor"
SPLIT_FILE = DATA / "splits.csv"

GRIDS = {
    "poly_ridge": [{"degree": d, "alpha": a} for d in (2, 3) for a in (1e-8, 1e-5, 1e-2)],
    "random_forest": [{"n_estimators": 300, "max_features": m, "min_samples_leaf": l}
                      for m in (0.33, 0.66, 1.0) for l in (1, 3)],
    "lightgbm": [{"n_estimators": n, "learning_rate": lr, "num_leaves": nl}
                 for n in (1500, 4000) for lr in (0.03, 0.08) for nl in (15, 63)],
    "mlp": [{"hidden": h, "alpha": a, "activation": "tanh", "solver": "lbfgs", "max_iter": 3000}
            for h in ((64, 64), (128, 128), (64, 64, 64), (128, 128, 128)) for a in (1e-7, 1e-5)],
}
QUICK_GRIDS = {
    "poly_ridge": [{"degree": 2, "alpha": 1e-5}],
    "random_forest": [{"n_estimators": 50, "max_features": 0.66, "min_samples_leaf": 1}],
    "lightgbm": [{"n_estimators": 200, "learning_rate": 0.1, "num_leaves": 31}],
    "mlp": [{"hidden": (32, 32), "alpha": 1e-5, "activation": "tanh", "solver": "lbfgs", "max_iter": 200}],
}
STUDY_LGBM = {"n_estimators": 1500, "learning_rate": 0.05, "num_leaves": 31}
STUDY_TRANSFORMS = [("linear", 1.0), ("asinh", 1e-18), ("asinh", 1e-15), ("asinh", 1e-12), ("asinh", 1e-9)]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ===================================================================== data + split
def load_case(case: str) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"{case}.csv")
    df.insert(0, "sample_id", pd.read_csv(DATA / f"{case}_index.csv").sample_id.to_numpy())
    return df


def make_or_load_splits(cases) -> pd.DataFrame:
    """70/15/15 per case. Seeded per case; saved once and re-used (verified)."""
    fr = CFG.split
    rows = []
    for k, case in enumerate(cases):
        ids = load_case(case).sample_id.to_numpy()
        rng = np.random.default_rng([CFG.seeds["split"], k])
        perm = rng.permutation(len(ids))
        n_tr, n_va = round(fr["train"] * len(ids)), round(fr["val"] * len(ids))
        lab = np.empty(len(ids), dtype=object)
        lab[perm[:n_tr]], lab[perm[n_tr:n_tr + n_va]], lab[perm[n_tr + n_va:]] = "train", "val", "test"
        rows.append(pd.DataFrame({"case": case, "sample_id": ids, "split": lab}))
    new = pd.concat(rows, ignore_index=True)
    if SPLIT_FILE.exists():
        old = pd.read_csv(SPLIT_FILE)
        if not old.equals(new):
            raise SystemExit(f"{SPLIT_FILE} differs from the seeded split; refusing to continue")
        return old
    new.to_csv(SPLIT_FILE, index=False)
    return new


def train_val(case: str, splits: pd.DataFrame):
    """Training and validation rows only. Test rows are dropped here, before anything else."""
    df = load_case(case)
    lab = splits[splits.case == case].set_index("sample_id").split
    s = lab.loc[df.sample_id].to_numpy()
    return df[s == "train"].reset_index(drop=True), df[s == "val"].reset_index(drop=True)


# ===================================================================== fitting
# Every fit (tuning and final refit) uses exactly this many BLAS/OpenMP threads.
# Floating-point summation order depends on the thread count, and over thousands of
# L-BFGS iterations a different count gives a (slightly) different MLP. Pinning it
# makes the saved models reproduce the tuned validation scores exactly.
FIT_THREADS = 2


def fit_model(case, algo, params, transform: TargetTransform, tr) -> CurrentModel:
    from sklearn.exceptions import ConvergenceWarning
    from threadpoolctl import threadpool_limits
    dev = CFG.cases[case].device
    with warnings.catch_warnings(), threadpool_limits(limits=FIT_THREADS):
        warnings.simplefilter("ignore", ConvergenceWarning)
        return CurrentModel(case, dev, algo, dict(params), nominal_process(CFG, dev), transform,
                            seed=CFG.seeds["training"]).fit(tr, n_jobs=FIT_THREADS)


def fit_and_score(case, algo, params, transform: TargetTransform, tr, va) -> dict:
    t0 = time.perf_counter()
    m = fit_model(case, algo, params, transform, tr)
    fit_s = time.perf_counter() - t0
    met = evaluate(va, m.predict(va))
    n_iter = getattr(m.estimator, "n_iter_", None)
    return {"case": case, "algo": algo, "params": params, "transform": transform.name,
            "score": selection_score(met), "fit_seconds": round(fit_s, 2),
            "n_iter": int(n_iter) if n_iter is not None else None,
            "metrics": met.to_dict(orient="records")}


def _limit_threads():
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "2"


def run_jobs(jobs, cache_dir: Path, splits, transform, n_workers, label):
    """jobs: list of (case, algo, params). Cached results are re-used."""
    from joblib import Parallel, delayed
    cache_dir.mkdir(parents=True, exist_ok=True)
    todo, results = [], []
    for case, algo, params in jobs:
        f = cache_dir / f"{case}__{algo}__{transform.name.replace('/', '_')}__{params_key(algo, params)}.json"
        if f.exists():
            results.append(json.loads(f.read_text(encoding="utf-8")))
        else:
            todo.append((case, algo, params, f))
    print(f"[{label}] {len(results)} cached, {len(todo)} to fit with {n_workers} workers")
    data = {c: train_val(c, splits) for c in {j[0] for j in todo}}

    def one(case, algo, params, f):
        _limit_threads()
        tr, va = data[case]
        r = fit_and_score(case, algo, params, transform, tr, va)
        f.write_text(json.dumps(r, indent=1, default=str), encoding="utf-8")
        return r

    t0 = time.perf_counter()
    done = Parallel(n_jobs=n_workers, backend="loky", verbose=0, return_as="generator_unordered")(
        delayed(one)(*j) for j in todo)
    for i, r in enumerate(done, 1):
        results.append(r)
        print(f"  [{label}] {i}/{len(todo)} {r['case']:8s} {r['algo']:13s} score={r['score']:.4g} "
              f"fit={r['fit_seconds']:.0f}s  ({time.perf_counter() - t0:.0f}s elapsed)", flush=True)
    return results


def flat(results) -> pd.DataFrame:
    rows = []
    for r in results:
        met = pd.DataFrame(r["metrics"]).set_index("terminal")
        row = {"case": r["case"], "algo": r["algo"], "transform": r["transform"],
               "params": json.dumps(r["params"], default=str), "score": r["score"],
               "fit_seconds": r["fit_seconds"], "n_iter": r.get("n_iter")}
        for t in TARGETS + ["mean"]:
            for k in ("r2_t", "med_rel", "p95_rel", "sign_acc"):
                row[f"{t}_{k}"] = met.loc[t, k]
        rows.append(row)
    return pd.DataFrame(rows)


# ===================================================================== sign+log variant (study only)
def signlog_study(case, tr, va) -> dict:
    """Alternative to asinh: LightGBM regressor on log10|I| plus a LightGBM sign
    classifier per terminal. Used only in the transform study."""
    from lightgbm import LGBMClassifier, LGBMRegressor
    from leakage.ml import build_features
    dev = CFG.cases[case].device
    nom = nominal_process(CFG, dev)
    Xtr, Xva = build_features(tr, dev, nom), build_features(va, dev, nom)
    pred = pd.DataFrame(index=va.index)
    for t in TARGETS:
        y = tr[t].to_numpy(float)
        reg = LGBMRegressor(**STUDY_LGBM, random_state=CFG.seeds["training"], verbose=-1, n_jobs=4)
        reg.fit(Xtr, np.log10(np.maximum(np.abs(y), 1e-22)))
        mag = 10 ** reg.predict(Xva)
        pos = y > 0
        if pos.all() or (~pos).all():
            sgn = np.full(len(va), 1.0 if pos.all() else -1.0)
        else:
            clf = LGBMClassifier(**STUDY_LGBM, random_state=CFG.seeds["training"], verbose=-1, n_jobs=4)
            clf.fit(Xtr, pos)
            sgn = np.where(clf.predict(Xva), 1.0, -1.0)
        pred[t] = sgn * mag
    met = evaluate(va, pred)
    return {"case": case, "algo": "lightgbm", "params": STUDY_LGBM, "transform": "sign+log10|I|",
            "score": selection_score(met), "fit_seconds": None, "metrics": met.to_dict(orient="records")}


# ===================================================================== main
def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, timeout=20).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def md(df: pd.DataFrame, fmt="{:.4g}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join((fmt.format(v) if isinstance(v, (float, np.floating)) else str(v))
                                     .replace("|", "\\|") for v in r) + " |")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny grid for a fast code check")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--cases", nargs="+", default=list(CFG.cases))
    a = ap.parse_args()

    out = CFG.results / ("a2_quick" if a.quick else "a2")
    out.mkdir(parents=True, exist_ok=True)
    splits = make_or_load_splits(list(CFG.cases))
    counts = splits.groupby(["case", "split"]).size().unstack()
    print("Split sizes:\n" + counts.to_string())

    # ---------------------------------------------------------------- 2. transform study
    t_all = time.perf_counter()
    transforms = STUDY_TRANSFORMS[:3] if a.quick else STUDY_TRANSFORMS
    study = []
    for kind, i0 in transforms:
        tf = TargetTransform(kind, i0)
        study += run_jobs([(c, "lightgbm", dict(STUDY_LGBM)) for c in a.cases],
                          out / "transform_study", splits, tf, a.workers, f"transform {tf.name}")
    for c in a.cases:
        tr, va = train_val(c, splits)
        study.append(signlog_study(c, tr, va))
    st = flat(study)
    st_tab = st.pivot_table(index="transform", columns="case", values="score").assign(
        mean_over_cases=lambda d: d.mean(axis=1)).sort_values("mean_over_cases")
    st.to_csv(out / "transform_study.csv", index=False)
    print("\nTransform study (validation mean-over-terminals median relative error, LightGBM):")
    print(st_tab.to_string(float_format=lambda x: f"{x:.4g}"))
    # Rule: choose the best SINGLE-MODEL transform (usable by every algorithm as one
    # multi-output regressor). sign+log10|I| needs an extra sign classifier per terminal,
    # so it is reported as an investigated alternative, not selected.
    single = {TargetTransform(k, i).name: TargetTransform(k, i) for k, i in transforms}
    ranked = [n for n in st_tab.index if n in single]
    chosen = single[ranked[0]]
    if st_tab.index[0] not in single:
        gap = st_tab.loc[ranked[0], "mean_over_cases"] - st_tab.iloc[0]["mean_over_cases"]
        print(f"Note: '{st_tab.index[0]}' scored best overall (by {gap:.3g}) but needs a separate "
              "sign classifier; choosing the best single-model transform.")
    print(f"Chosen target transform: {chosen.name}")

    # ---------------------------------------------------------------- 3. tuning
    grids = QUICK_GRIDS if a.quick else GRIDS
    jobs = [(c, algo, p) for c in a.cases for algo in ALGORITHMS for p in grids[algo]]
    res = run_jobs(jobs, out / "tuning", splits, chosen, a.workers, "tuning")
    tun = flat(res)
    tun = tun[tun["transform"] == chosen.name]
    tun.to_csv(out / "tuning_all.csv", index=False)

    # ---------------------------------------------------------------- 4. comparison + selection
    best = tun.loc[tun.groupby(["case", "algo"]).score.idxmin()].sort_values(["case", "score"])
    comp = best[["case", "algo", "params", "mean_r2_t", "mean_med_rel", "mean_p95_rel",
                 *[f"{t}_{k}" for t in TARGETS for k in ("r2_t", "med_rel", "p95_rel")], "fit_seconds"]]
    comp.to_csv(out / "validation_comparison.csv", index=False)
    sel = best.loc[best.groupby("case").score.idxmin()].set_index("case")

    # ---------------------------------------------------------------- 5. refit + save
    saved = {}
    if not a.quick:
        CFG.trained_models.mkdir(parents=True, exist_ok=True)
        import joblib
        import sklearn
        from joblib import Parallel, delayed

        def sel_params(case):
            p = json.loads(sel.loc[case].params)
            if "hidden" in p:
                p["hidden"] = tuple(p["hidden"])
            return p

        print(f"\nRefitting {len(a.cases)} selected models in parallel ({FIT_THREADS} threads each) ...")
        t_refit = time.perf_counter()
        models = Parallel(n_jobs=min(len(a.cases), a.workers), backend="loky")(
            delayed(fit_model)(c, sel.loc[c].algo, sel_params(c), chosen, train_val(c, splits)[0])
            for c in a.cases)
        print(f"refit done in {time.perf_counter() - t_refit:.0f}s")
        for case, model in zip(a.cases, models):
            r = sel.loc[case]
            params = sel_params(case)
            tr, va = train_val(case, splits)
            dev = CFG.cases[case].device
            val_met = evaluate(va, model.predict(va))
            reproduced = bool(np.isclose(selection_score(val_met), r.score, rtol=1e-9, atol=0))
            print(f"{case}: refit validation score {selection_score(val_met):.8g} vs tuned {r.score:.8g} "
                  f"-> {'reproduced exactly' if reproduced else 'MISMATCH'}")
            model.meta = {
                "created": dt.datetime.now().isoformat(timespec="seconds"),
                "case": case, "device": dev,
                "case_rule": {"vg_frac": CFG.cases[case].vg_frac, "vb_frac": CFG.cases[case].vb_frac},
                "algorithm": r.algo, "params": params, "transform": chosen.name,
                "features": FEATURES, "targets": TARGETS, "nominal_process": nominal_process(CFG, dev),
                "trained_on": "training split only (70%)", "n_train": len(tr), "n_val": len(va),
                "split_seed": CFG.seeds["split"], "training_seed": CFG.seeds["training"],
                "split_file_sha256": sha256(SPLIT_FILE),
                "dataset_sha256": sha256(DATA / f"{case}.csv"),
                "spice_model": CFG.spice_model_file.name, "spice_model_sha256": CFG.spice_model_sha256,
                "lmin": CFG.lmin, "vdd_nominal": CFG.vdd_nominal,
                "input_ranges": {"vdd": CFG.vdd_range, "temp": CFG.temp_range, "w": CFG.w_range,
                                 "process_tolerance": CFG.process_tolerance},
                "validation_metrics": val_met.to_dict(orient="records"),
                "validation_score": selection_score(val_met),
                "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                             "numpy": np.__version__, "pandas": pd.__version__},
                "git_commit": git_commit(), "fit_threads": FIT_THREADS,
                "refit_reproduces_tuning": reproduced,
            }
            path = CFG.trained_models / f"{case}.joblib"
            joblib.dump(model, path, compress=3)
            saved[case] = {"path": str(path.relative_to(ROOT)), "sha256": sha256(path),
                           "algorithm": r.algo, "params": params, "validation_score": selection_score(val_met)}
            print(f"saved {path.name}: {r.algo} {params}  val score {selection_score(val_met):.4g}")
        (out / "selection.json").write_text(json.dumps(
            {"transform": chosen.name, "selection_criterion": "mean over Id/Ig/Is/Ib of validation median "
             "relative error", "models": saved}, indent=2, default=str), encoding="utf-8")

    # ---------------------------------------------------------------- report
    pd.set_option("display.width", 250)
    show = comp[["case", "algo", "mean_r2_t", "mean_med_rel", "mean_p95_rel",
                 "id_med_rel", "ig_med_rel", "is_med_rel", "ib_med_rel", "fit_seconds"]]
    print("\nValidation comparison (best config per algorithm):")
    print(show.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print("\nSelected per case:\n" + sel[["algo", "params", "score"]].to_string())

    lines = ["# Part A2: validation comparison", "",
             f"Generated {dt.datetime.now().isoformat(timespec='seconds')}. Train/val only; the test split was not used.",
             "", f"Split sizes per case: {counts.iloc[0].to_dict()}", "",
             "## Transform study (LightGBM, fixed settings)", "",
             "Value = validation mean over the four terminals of the median relative error.", "",
             md(st_tab.reset_index()), "", f"**Chosen transform: {chosen.name}**", "",
             "## Four algorithms x four cases (best configuration of each algorithm)", "",
             "R2_t = R^2 of asinh(I/1 fA); med/p95 = median/95th-percentile relative error "
             "|I_pred - I_spice| / max(|I_spice|, 1 fA).", "",
             md(comp.drop(columns=["params"]).rename(columns=lambda c: c.replace("mean_", "avg_"))), "",
             "## Selected models (lowest validation score)", "",
             md(sel[["algo", "params", "score"]].reset_index()), "",
             "## Best hyperparameters per algorithm", "", md(comp[["case", "algo", "params"]]), ""]
    (out / "validation_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nTotal {time.perf_counter() - t_all:.0f}s. Outputs in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
