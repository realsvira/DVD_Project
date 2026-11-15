"""Part A2b: improve the transistor models where v1 was weakest (Id/Is at small |Vds|
and near the 0/VDD rails), using validation data only for every decision.

Data
  * original spec datasets: the existing 70/15/15 split (data/transistor/splits.csv);
    the original TEST rows are not loaded here.
  * supplementary targeted samples (scripts/a1_generate_datasets.py --profile targeted
    --seed-salt 1): data/transistor_aug_on (4000/case) and data/transistor_aug_off
    (2000/case), split 80/20 train/val with a fixed seed (data/transistor_aug_splits.csv).
Model
  * MLP only: in Part A2 it beat LightGBM / random forest / polynomial ridge by 3-5x in
    every case, so the algorithm comparison is not repeated.
  * feature set v2 = v1 + asinh(Vds / 1 uV); same target transform asinh(I/1e-15).
  * grid: (128,128,128) alpha 1e-7 / 1e-5, (192,192,192) alpha 1e-7; 2 BLAS threads per fit.
Selection (validation only)
  score = mean( selection_score(original val), selection_score(targeted val) ),
  selection_score = mean over Id/Ig/Is/Ib of the median relative error.
  For each case the better of {v1 model, best v2 model} is kept.
Outputs
  trained_models/v1/<case>.joblib   (archived Part A2 models)
  trained_models/v2/<case>.joblib   (best v2 model per case)
  trained_models/<case>.joblib      (FINAL model per case = selected one)
  results/a2b/validation_v1_vs_v2.csv/.md, results/a2b/selection.json

Run:  .venv/bin/python scripts/a2b_train_v2.py
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from leakage.config import load_config  # noqa: E402
from leakage.ml import (FEATURE_SETS, INPUTS, TARGETS, CurrentModel, TargetTransform,  # noqa: E402
                        evaluate, selection_score)
from leakage.transistor import nominal_process  # noqa: E402

CFG = load_config()
DATA = CFG.data
OUT = CFG.results / "a2b"
FIT_THREADS = 2
AUG_DIRS = {"nmos_on": "transistor_aug_on", "pmos_on": "transistor_aug_on",
            "nmos_off": "transistor_aug_off", "pmos_off": "transistor_aug_off"}
AUG_SPLIT_FILE = DATA / "transistor_aug_splits.csv"
GRID = [{"hidden": (128, 128, 128), "alpha": 1e-7}, {"hidden": (128, 128, 128), "alpha": 1e-5},
        {"hidden": (192, 192, 192), "alpha": 1e-7}]
SMALL_VDS = 0.05     # V


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(dirname: str, case: str) -> pd.DataFrame:
    d = DATA / dirname
    df = pd.read_csv(d / f"{case}.csv")
    df.insert(0, "sample_id", pd.read_csv(d / f"{case}_index.csv").sample_id.to_numpy())
    return df


def original_train_val(case):
    df = load("transistor", case)
    sp = pd.read_csv(DATA / "transistor" / "splits.csv")
    lab = sp[sp.case == case].set_index("sample_id").split.loc[df.sample_id].to_numpy()
    return df[lab == "train"].reset_index(drop=True), df[lab == "val"].reset_index(drop=True)


def aug_splits() -> pd.DataFrame:
    rows = []
    for k, case in enumerate(CFG.cases):
        ids = load(AUG_DIRS[case], case).sample_id.to_numpy()
        perm = np.random.default_rng([CFG.seeds["split"], k, 1]).permutation(len(ids))
        lab = np.full(len(ids), "train", dtype=object)
        lab[perm[: round(0.2 * len(ids))]] = "val"
        rows.append(pd.DataFrame({"case": case, "sample_id": ids, "split": lab}))
    new = pd.concat(rows, ignore_index=True)
    if AUG_SPLIT_FILE.exists():
        old = pd.read_csv(AUG_SPLIT_FILE)
        if not old.equals(new):
            raise SystemExit(f"{AUG_SPLIT_FILE} differs from the seeded split")
        return old
    new.to_csv(AUG_SPLIT_FILE, index=False)
    return new


def aug_train_val(case, sp):
    df = load(AUG_DIRS[case], case)
    lab = sp[sp.case == case].set_index("sample_id").split.loc[df.sample_id].to_numpy()
    return df[lab == "train"].reset_index(drop=True), df[lab == "val"].reset_index(drop=True)


def fit_v2(case, params, train):
    from sklearn.exceptions import ConvergenceWarning
    from threadpoolctl import threadpool_limits
    dev = CFG.cases[case].device
    p = {**params, "activation": "tanh", "solver": "lbfgs", "max_iter": 3000}
    t0 = time.perf_counter()
    with warnings.catch_warnings(), threadpool_limits(limits=FIT_THREADS):
        warnings.simplefilter("ignore", ConvergenceWarning)
        m = CurrentModel(case, dev, "mlp", p, nominal_process(CFG, dev), TargetTransform("asinh", 1e-15),
                         seed=CFG.seeds["training"], feature_set="v2").fit(train, n_jobs=FIT_THREADS)
    return m, time.perf_counter() - t0


def scores(model, va_orig, va_aug) -> dict:
    out = {}
    for name, va in (("val_orig", va_orig), ("val_targeted", va_aug)):
        met = evaluate(va, model.predict(va[INPUTS]))
        out[name] = selection_score(met)
        for t in TARGETS:
            out[f"{name}_{t}_med_rel"] = float(met.set_index("terminal").loc[t, "med_rel"])
    both = pd.concat([va_orig, va_aug], ignore_index=True)
    small = both[(both.vd - both.vs).abs() < SMALL_VDS]
    met = evaluate(small, model.predict(small[INPUTS]))
    for t in ("id", "is"):
        out[f"val_smallvds_{t}_med_rel"] = float(met.set_index("terminal").loc[t, "med_rel"])
    out["score"] = 0.5 * (out["val_orig"] + out["val_targeted"])
    return out


def md(df: pd.DataFrame, fmt="{:.4g}") -> str:
    cols = list(df.columns)
    rows = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        rows.append("| " + " | ".join((fmt.format(v) if isinstance(v, (float, np.floating)) else str(v))
                                      .replace("|", "\\|") for v in r) + " |")
    return "\n".join(rows)


def main() -> int:
    import joblib
    from joblib import Parallel, delayed
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "model_cache"
    cache.mkdir(exist_ok=True)
    sp = aug_splits()
    print(sp.groupby(["case", "split"]).size().unstack().to_string())

    # archive v1 once
    v1_dir = CFG.trained_models / "v1"
    v1_dir.mkdir(parents=True, exist_ok=True)
    for case in CFG.cases:
        if not (v1_dir / f"{case}.joblib").exists():
            shutil.copy2(CFG.trained_models / f"{case}.joblib", v1_dir / f"{case}.joblib")

    data = {}
    for case in CFG.cases:
        tr, va = original_train_val(case)
        atr, ava = aug_train_val(case, sp)
        data[case] = (pd.concat([tr, atr], ignore_index=True), va, ava)

    jobs = [(c, i, p) for c in CFG.cases for i, p in enumerate(GRID)]
    todo = [j for j in jobs if not (cache / f"{j[0]}__{j[1]}.joblib").exists()]
    print(f"{len(jobs) - len(todo)} cached, {len(todo)} MLP fits to run")

    def one(case, i, params):
        m, secs = fit_v2(case, params, data[case][0])
        m.meta = {"fit_seconds": round(secs, 1), "grid_index": i}
        joblib.dump(m, cache / f"{case}__{i}.joblib", compress=3)
        return case, i, secs

    t0 = time.perf_counter()
    for case, i, secs in Parallel(n_jobs=6, backend="loky", return_as="generator_unordered")(
            delayed(one)(*j) for j in todo):
        print(f"  fitted {case} config {i} in {secs:.0f}s ({time.perf_counter() - t0:.0f}s elapsed)", flush=True)

    rows, best = [], {}
    for case in CFG.cases:
        _, va, ava = data[case]
        v1 = joblib.load(v1_dir / f"{case}.joblib")
        rows.append({"case": case, "model": "v1 (A2)", "params": json.dumps(v1.params, default=str),
                     **scores(v1, va, ava)})
        for i, p in enumerate(GRID):
            m = joblib.load(cache / f"{case}__{i}.joblib")
            r = {"case": case, "model": f"v2 #{i}", "params": json.dumps(p, default=str), **scores(m, va, ava)}
            rows.append(r)
            if case not in best or r["score"] < best[case][1]["score"]:
                best[case] = (m, r)
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "validation_v1_vs_v2.csv", index=False)

    v2_dir = CFG.trained_models / "v2"
    v2_dir.mkdir(exist_ok=True)
    selection = {}
    for case in CFG.cases:
        m, r = best[case]
        tr = data[case][0]
        m.meta = {**m.meta,
                  "created": dt.datetime.now().isoformat(timespec="seconds"), "case": case,
                  "device": m.device, "algorithm": "mlp", "params": m.params, "feature_set": "v2",
                  "features": FEATURE_SETS["v2"], "transform": m.transform.name,
                  "trained_on": "original 70% train split + 80% of targeted supplementary samples",
                  "n_train": len(tr), "fit_threads": FIT_THREADS, "training_seed": CFG.seeds["training"],
                  "split_file_sha256": sha256(DATA / "transistor" / "splits.csv"),
                  "aug_split_file_sha256": sha256(AUG_SPLIT_FILE),
                  "dataset_sha256": sha256(DATA / "transistor" / f"{case}.csv"),
                  "aug_dataset_sha256": sha256(DATA / AUG_DIRS[case] / f"{case}.csv"),
                  "validation": {k: v for k, v in r.items() if k not in ("case", "model", "params")}}
        joblib.dump(m, v2_dir / f"{case}.joblib", compress=3)
        v1_score = tab[(tab.case == case) & (tab.model == "v1 (A2)")].score.iloc[0]
        winner = "v2" if r["score"] < v1_score else "v1"
        src = (v2_dir if winner == "v2" else v1_dir) / f"{case}.joblib"
        shutil.copy2(src, CFG.trained_models / f"{case}.joblib")
        selection[case] = {"selected": winner, "v1_score": v1_score, "v2_score": r["score"],
                           "v2_config": r["model"], "file_sha256": sha256(CFG.trained_models / f"{case}.joblib")}
        print(f"{case}: v1 score {v1_score:.4f}  best v2 ({r['model']}) {r['score']:.4f}  -> {winner}")
    (OUT / "selection.json").write_text(json.dumps(
        {"criterion": "mean of [original-val score, targeted-val score]; score = mean over 4 terminals of "
         "median relative error", "cases": selection}, indent=2, default=str), encoding="utf-8")

    show = tab[["case", "model", "score", "val_orig", "val_targeted", "val_smallvds_id_med_rel",
                "val_smallvds_is_med_rel"]]
    pd.set_option("display.width", 220)
    print(show.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    (OUT / "validation_v1_vs_v2.md").write_text(
        "# Part A2b: v1 vs v2 on validation\n\nScores = mean over Id/Ig/Is/Ib of median relative error. "
        "`val_orig` = original 15 % validation split (spec distribution); `val_targeted` = 20 % of the "
        "targeted supplementary samples; `val_smallvds_*` = |Vds| < 0.05 V rows of both.\n\n"
        + md(show) + "\n\n" + md(pd.DataFrame(selection).T.reset_index().rename(columns={"index": "case"}))
        + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
