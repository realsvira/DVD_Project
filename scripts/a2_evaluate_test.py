"""Part A2: ONE-TIME evaluation of the four selected models on the held-out test split.

Guards
  * Refuses to run if results/a2/test_evaluation_record.json already exists
    (the test set is evaluated exactly once). --force overrides and is recorded.
  * Verifies each saved model was trained on the current split file and dataset
    (sha256 in the model metadata) and that model files have not changed since
    training (sha256 in results/a2/selection.json).

Outputs (results/a2/)
  test_results.csv / .md            per case x terminal: R2, median/p95 relative error,
                                    absolute error, scale error, sign accuracy, fractions < 1 %/5 %
  test_predictions_<case>.csv       sample_id, inputs, SPICE and predicted currents
  fig_pred_vs_spice_<case>.png      predicted vs SPICE, one figure per case (4 panels)
  test_error_analysis.csv / .md     errors by terminal and by |I|, |Vds|, source offset,
                                    temperature bins; worst samples
  test_evaluation_record.json       timestamp, model hashes, headline numbers

Run:  .venv/bin/python scripts/a2_evaluate_test.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from leakage.config import load_config  # noqa: E402
from leakage.ml import I_FLOOR, INPUTS, TARGETS, evaluate  # noqa: E402

CFG = load_config()
DATA = CFG.data / "transistor"
OUT = CFG.results / "a2"
RECORD = OUT / "test_evaluation_record.json"


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def md(df: pd.DataFrame, fmt="{:.4g}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join((fmt.format(v) if isinstance(v, (float, np.floating)) else str(v))
                                     .replace("|", "\\|") for v in r) + " |")
    return "\n".join(out)


def test_rows(case: str, splits: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"{case}.csv")
    df.insert(0, "sample_id", pd.read_csv(DATA / f"{case}_index.csv").sample_id.to_numpy())
    lab = splits[splits.case == case].set_index("sample_id").split
    return df[lab.loc[df.sample_id].to_numpy() == "test"].reset_index(drop=True)


def plot_case(case, true, pred, met, path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, t in zip(axes.ravel(), TARGETS):
        a, b = true[t].to_numpy(), pred[t].to_numpy()
        agree = np.sign(a) == np.sign(b)
        x, y = np.maximum(np.abs(a), 1e-19), np.maximum(np.abs(b), 1e-19)
        ax.scatter(x[agree], y[agree], s=5, alpha=0.5, lw=0, c="#1f77b4", label=f"same sign (n={agree.sum()})")
        if (~agree).any():
            ax.scatter(x[~agree], y[~agree], s=12, alpha=0.9, lw=0, c="#d62728",
                       label=f"opposite sign (n={(~agree).sum()})")
        lo, hi = min(x.min(), y.min()) / 2, max(x.max(), y.max()) * 2
        ax.plot([lo, hi], [lo, hi], "k-", lw=0.8)
        ax.plot([lo, hi], [lo * 1.1, hi * 1.1], "k:", lw=0.6)
        ax.plot([lo, hi], [lo / 1.1, hi / 1.1], "k:", lw=0.6, label="±10 %")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        r = met.set_index("terminal").loc[t]
        ax.set_title(f"{t.upper()}:  median err {r.med_rel:.2%},  p95 {r.p95_rel:.1%},  R²ₜ {r.r2_t:.4f}",
                     fontsize=10)
        ax.set_xlabel(f"|{t}| SPICE (A)"); ax.set_ylabel(f"|{t}| ML prediction (A)")
        ax.grid(True, which="major", alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")
    fig.suptitle(f"{case.upper()}: ML prediction vs SPICE on the held-out test set (n={len(true)})", fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def binned(df, by, bins, labels):
    rows = []
    cat = pd.cut(df[by], bins=bins, labels=labels, include_lowest=True)
    for (term, b), g in df.groupby(["terminal", cat], observed=True):
        rows.append({"terminal": term, "by": by, "bin": str(b), "n": len(g),
                     "med_rel": g.rel.median(), "p95_rel": g.rel.quantile(0.95)})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-run although the test set was already evaluated")
    a = ap.parse_args()
    if RECORD.exists() and not a.force:
        print(f"The test set was already evaluated ({RECORD}). Refusing to evaluate again.\n"
              "Use --force only if you understand this repeats the one-time test evaluation.")
        return 2
    import joblib
    sel = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))
    splits = pd.read_csv(DATA / "splits.csv")
    split_hash = sha256(DATA / "splits.csv")

    all_met, err_rows, analysis, record = [], [], [], {"models": {}}
    for case in CFG.cases:
        path = CFG.trained_models / f"{case}.joblib"
        h = sha256(path)
        if h != sel["models"][case]["sha256"]:
            raise SystemExit(f"{path} changed after training/selection; refusing")
        m = joblib.load(path)
        if m.meta["split_file_sha256"] != split_hash or m.meta["dataset_sha256"] != sha256(DATA / f"{case}.csv"):
            raise SystemExit(f"{case}: model was trained on a different split or dataset")
        te = test_rows(case, splits)
        pred = m.predict(te[INPUTS])
        met = evaluate(te, pred)
        met.insert(0, "case", case); met.insert(1, "algorithm", m.algo)
        all_met.append(met)
        record["models"][case] = {"file_sha256": h, "algorithm": m.algo, "n_test": len(te)}

        out = te.copy()
        for t in TARGETS:
            out[f"{t}_pred"] = pred[t].to_numpy()
        out.to_csv(OUT / f"test_predictions_{case}.csv", index=False)
        plot_case(case, te, pred, met, OUT / f"fig_pred_vs_spice_{case}.png")

        # ----- error analysis
        dev = CFG.cases[case].device
        src_off = (te[["vd", "vs"]].min(axis=1) if dev == "nmos" else te.vdd - te[["vd", "vs"]].max(axis=1))
        long = []
        for t in TARGETS:
            e = np.abs(pred[t].to_numpy() - te[t].to_numpy())
            long.append(pd.DataFrame({"terminal": t, "abs_i": te[t].abs().to_numpy(),
                                      "rel": e / np.maximum(te[t].abs().to_numpy(), I_FLOOR),
                                      "avds": (te.vd - te.vs).abs().to_numpy(),
                                      "src_off": (src_off / te.vdd).to_numpy(), "temp": te.temp.to_numpy(),
                                      "sample_id": te.sample_id.to_numpy(),
                                      "sign_ok": np.sign(pred[t].to_numpy()) == np.sign(te[t].to_numpy())}))
        L = pd.concat(long, ignore_index=True)
        rows = []
        rows += binned(L, "abs_i", [0, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1],
                       ["<1e-14", "1e-14..1e-12", "1e-12..1e-10", "1e-10..1e-8", "1e-8..1e-6", ">1e-6"])
        rows += binned(L, "avds", [0, 0.05, 0.2, 0.5, 1.2], ["<0.05 V", "0.05-0.2 V", "0.2-0.5 V", ">0.5 V"])
        rows += binned(L, "src_off", [0, 0.05, 0.2, 0.5, 1.01], ["<0.05 VDD", "0.05-0.2", "0.2-0.5", ">0.5"])
        rows += binned(L, "temp", [-40, 15, 70, 125], ["-40..15 C", "15..70 C", "70..125 C"])
        a_df = pd.DataFrame(rows); a_df.insert(0, "case", case)
        analysis.append(a_df)
        worst = L.sort_values("rel", ascending=False).head(10).copy()
        worst.insert(0, "case", case)
        err_rows.append(worst)
        # KCL of predictions (not enforced by the model)
        kcl = pred[TARGETS].sum(axis=1).abs() / pred[TARGETS].abs().max(axis=1)
        record["models"][case]["pred_kcl_rel_median"] = float(kcl.median())
        record["models"][case]["pred_kcl_rel_p95"] = float(kcl.quantile(0.95))

    res = pd.concat(all_met, ignore_index=True)
    res.to_csv(OUT / "test_results.csv", index=False)
    an = pd.concat(analysis, ignore_index=True)
    an.to_csv(OUT / "test_error_analysis.csv", index=False)
    worst = pd.concat(err_rows, ignore_index=True)
    worst.to_csv(OUT / "test_worst_samples.csv", index=False)

    cols = ["case", "algorithm", "terminal", "r2_t", "r2_raw", "med_rel", "p95_rel", "frac_rel_lt_1pct",
            "frac_rel_lt_5pct", "med_abs", "p95_abs", "med_scale", "p95_scale", "sign_acc"]
    headline = res[res.terminal == "mean"][["case", "algorithm", "med_rel", "p95_rel", "r2_t"]]
    lines = ["# Part A2: test-set results (one-time evaluation)", "",
             f"Evaluated {dt.datetime.now().isoformat(timespec='seconds')} on the held-out 15 % test split "
             "(1,500 samples per case), never used for training, tuning or selection.", "",
             "Error definitions: rel = |I_pred − I_spice| / max(|I_spice|, 1 fA); abs in A; "
             "scale = |I_pred − I_spice| / max over the 4 terminals of |I_spice|; sign_acc counts samples with "
             "|I_spice| ≥ 10 fA; R2_t = R² of asinh(I/1 fA); R2_raw = R² on currents in A.", "",
             "## Headline (mean over the four terminals)", "", md(headline), "",
             "## Per case and terminal", "", md(res[cols]), "",
             "## Error vs conditions (median / p95 relative error)", "",
             md(an.pivot_table(index=["case", "by", "bin"], columns="terminal", values="med_rel")
                .reset_index()), "",
             "Full tables: `test_error_analysis.csv`, `test_worst_samples.csv`, `test_predictions_<case>.csv`.", ""]
    (OUT / "test_results.md").write_text("\n".join(lines), encoding="utf-8")

    record.update({"evaluated": dt.datetime.now().isoformat(timespec="seconds"), "forced": bool(a.force),
                   "split_file_sha256": split_hash,
                   "headline": headline.to_dict(orient="records")})
    RECORD.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")

    pd.set_option("display.width", 250)
    print(res[cols].to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print(f"\nRecord written to {RECORD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
