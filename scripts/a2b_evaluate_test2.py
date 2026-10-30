"""Part A2b: ONE-TIME evaluation of v1 and v2 models on FRESH test sets that no model,
tuning step or selection step has seen:

  data/transistor_test2_uniform   1500 / case, spec distribution, independent seed (salt 2)
  data/transistor_test2_targeted  1000 / case, near-rail / small-|Vds| profile (salt 3)

The original test split was already used once (Part A2) and is not reused for decisions.

Outputs (results/a2b/)
  test2_results.csv / .md            case x model x test set x terminal metrics
  test2_smallvds.csv                 Id/Is error for |Vds| < 0.05 V
  fig_pred_vs_spice_final_<case>.png          final model, uniform fresh test set
  fig_pred_vs_spice_final_targeted_<case>.png final model, targeted fresh test set
  test2_evaluation_record.json       guard + model hashes

Run:  .\\.venv\\Scripts\\python.exe scripts\\a2b_evaluate_test2.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from a2_evaluate_test import md, plot_case  # noqa: E402
from leakage.config import load_config  # noqa: E402
from leakage.ml import INPUTS, TARGETS, evaluate  # noqa: E402

CFG = load_config()
OUT = CFG.results / "a2b"
RECORD = OUT / "test2_evaluation_record.json"
TESTS = {"uniform": "transistor_test2_uniform", "targeted": "transistor_test2_targeted"}


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(dirname, case):
    d = CFG.data / dirname
    df = pd.read_csv(d / f"{case}.csv")
    df.insert(0, "sample_id", pd.read_csv(d / f"{case}_index.csv").sample_id.to_numpy())
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if RECORD.exists() and not a.force:
        print(f"Fresh test sets already evaluated ({RECORD}); refusing to evaluate again.")
        return 2
    import joblib
    sel = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))["cases"]
    rows, small_rows, record = [], [], {"models": {}}
    for case in CFG.cases:
        final_path = CFG.trained_models / f"{case}.joblib"
        if sha256(final_path) != sel[case]["file_sha256"]:
            raise SystemExit(f"{final_path} changed after selection; refusing")
        models = {"v1": joblib.load(CFG.trained_models / "v1" / f"{case}.joblib"),
                  "v2": joblib.load(CFG.trained_models / "v2" / f"{case}.joblib")}
        final_tag = sel[case]["selected"]
        record["models"][case] = {"final": final_tag, "final_sha256": sha256(final_path)}
        for tname, dname in TESTS.items():
            te = load(dname, case)
            for tag, m in models.items():
                pred = m.predict(te[INPUTS])
                met = evaluate(te, pred)
                met.insert(0, "case", case); met.insert(1, "model", tag); met.insert(2, "test_set", tname)
                met.insert(3, "final", tag == final_tag)
                rows.append(met)
                sm = (te.vd - te.vs).abs() < 0.05
                if sm.any():
                    ms = evaluate(te[sm], pred[sm]).set_index("terminal")
                    small_rows.append({"case": case, "model": tag, "test_set": tname, "n": int(sm.sum()),
                                       **{f"{t}_med_rel": ms.loc[t, "med_rel"] for t in TARGETS},
                                       **{f"{t}_p95_rel": ms.loc[t, "p95_rel"] for t in ("id", "is")}})
                if tag == final_tag:
                    suffix = "" if tname == "uniform" else "_targeted"
                    plot_case(f"{case} [{tag}, fresh {tname} test]", te, pred, met.drop(columns=["case", "model", "test_set", "final"]),
                              OUT / f"fig_pred_vs_spice_final{suffix}_{case}.png")
    res = pd.concat(rows, ignore_index=True)
    res.to_csv(OUT / "test2_results.csv", index=False)
    small = pd.DataFrame(small_rows)
    small.to_csv(OUT / "test2_smallvds.csv", index=False)

    head = res[res.terminal == "mean"][["case", "model", "final", "test_set", "med_rel", "p95_rel", "r2_t"]]
    per = res[(res.terminal != "mean")].pivot_table(index=["case", "test_set", "terminal"], columns="model",
                                                    values="med_rel").reset_index()
    lines = ["# Part A2b: fresh test-set results (one-time evaluation)", "",
             f"Evaluated {dt.datetime.now().isoformat(timespec='seconds')}. Both test sets were generated with "
             "independent seeds after Part A2 and were not used for training, tuning or selection.", "",
             "## Mean over Id/Ig/Is/Ib", "", md(head), "",
             "## Median relative error per terminal (v1 vs v2)", "", md(per), "",
             "## |Vds| < 0.05 V (median relative error)", "", md(small), ""]
    (OUT / "test2_results.md").write_text("\n".join(lines), encoding="utf-8")
    record.update({"evaluated": dt.datetime.now().isoformat(timespec="seconds"), "forced": bool(a.force),
                   "headline": head.to_dict(orient="records")})
    RECORD.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    pd.set_option("display.width", 220)
    print(head.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    print(small.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
