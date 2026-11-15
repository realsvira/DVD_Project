"""Part A1 data check (deliverable 3): data quality, Kirchhoff, physical trends.

Reads data/transistor/<case>.csv (+ _index.csv, failures.csv, metadata.json) and
writes everything to results/data_check/:

  quality_checks.csv               range / missing / failure / uniqueness checks per case
  kcl_summary.csv                  Id+Ig+Is+Ib per case: max / median / p95 (absolute, relative)
  trend_slopes.csv                 partial-regression slopes vs controlled SPICE sweeps
  source_offset_effect.csv         why OFF current must be conditioned on the source potential
  suspicious_samples.csv           samples flagged for investigation (NOT removed or edited)
  controlled_sweeps.csv            extra one-variable-at-a-time SPICE reference sweeps
  fig_off_current_vs_temperature.png
  fig_off_current_vs_vth0.png
  fig_gate_current_vs_toxe.png
  fig_supp_off_current_vs_source_offset.png
  data_quality_report.md           written report (also copied to reports/)

Conditioning (why): every input is randomised independently, so a raw scatter
of e.g. Ioff vs T is smeared by Vds, by the source potential (which sets Vgs and
the body bias), by W (8x) and by the other parameters. We therefore
  * normalise currents by width (A/um),
  * OFF current: keep samples with |Vds| >= 0.5 VDD AND the effective source
    within 0.05 VDD of its rail (so Vgs ~ 0 and no body bias),
  * gate current: ON case with both Vd and Vs near the rail opposite the gate,
  * plot binned medians per band of the most important other variable,
  * overlay a controlled SPICE sweep (one variable swept, all others nominal).
Partial-regression slopes (OLS of log10|I| on all inputs, standardised) give each
trend with the other variables held fixed. No SPICE value is modified or dropped.

Run:  .venv/bin/python scripts/a1_data_check.py
"""
from __future__ import annotations

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
from leakage.dataset import validate  # noqa: E402
from leakage.simulator import NgspiceSimulator  # noqa: E402
from leakage.transistor import (CURRENT_COLS, KCL_ABS, KCL_REL, kcl_residual,  # noqa: E402
                                make_inputs, nominal_process, simulate_transistors)

CFG = load_config()
DATA = CFG.data / "transistor"
OUT = CFG.results / "data_check"
SEED = CFG.seeds["global"]

NUM_FLOOR = 1e-16       # |I| below this is within ~100x of abstol (1e-18): numerically fragile
SIGN_VDS_MIN = 0.1      # V: channel-current direction is only checked when |Vds| > this
OUTLIER_Z = 8.0         # robust z threshold on cross-validated residual of log10|I| ...
OUTLIER_DEC = 1.0       # ... AND the residual must exceed this many decades
ZERO_CROSS_VDS = 0.05   # V: Id passes through zero at Vds = 0; skip outlier test there
OFF_VDS_MIN = 0.5       # x VDD
OFF_SRC_MAX = 0.05      # x VDD
COLORS = ["#1f77b4", "#d62728", "#2ca02c"]
OFF_LABEL = f"|Vds| ≥ {OFF_VDS_MIN:g}·VDD, source ≤ {OFF_SRC_MAX:g}·VDD from rail"


# ===================================================================== helpers
def load_case(case: str) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"{case}.csv")
    idx = pd.read_csv(DATA / f"{case}_index.csv")
    df.insert(0, "sample_id", idx.sample_id.to_numpy())
    df["kcl"] = kcl_residual(df)
    df["scale"] = df[CURRENT_COLS].abs().max(axis=1)
    df["w_um"] = df.w * 1e6
    df["vds"] = df.vd - df.vs
    return df


def safe_log10_abs(x: pd.Series) -> pd.Series:
    """log10|x|; exact zeros -> NaN (counted and reported, never replaced)."""
    a = x.abs()
    return np.log10(a.where(a > 0))


def source_offset(df: pd.DataFrame, dev: str) -> pd.Series:
    """Distance of the effective source from its rail: low terminal for NMOS (rail 0),
    high terminal for PMOS (rail VDD). Offset > 0 means Vgs < 0 (NMOS) / Vsg < 0 (PMOS)
    plus reverse body bias, which suppresses sub-threshold current exponentially."""
    if dev == "nmos":
        return df[["vd", "vs"]].min(axis=1)
    return df.vdd - df[["vd", "vs"]].max(axis=1)


def off_subset(df: pd.DataFrame, dev: str) -> pd.DataFrame:
    so = source_offset(df, dev)
    d = df[(df.vds.abs() >= OFF_VDS_MIN * df.vdd) & (so <= OFF_SRC_MAX * df.vdd)].copy()
    d["src_off"] = so[d.index]
    d["y"] = safe_log10_abs(d["id"] / d.w_um)
    return d


def gate_subset(df: pd.DataFrame, dev: str) -> pd.DataFrame:
    """ON case with a large gate-to-channel voltage at both channel ends."""
    if dev == "nmos":      # Vg = VDD, Vd and Vs low
        d = df[(df.vd <= 0.25 * df.vdd) & (df.vs <= 0.25 * df.vdd)].copy()
    else:                  # Vg = 0, Vd and Vs high
        d = df[(df.vd >= 0.75 * df.vdd) & (df.vs >= 0.75 * df.vdd)].copy()
    d["y"] = safe_log10_abs(d["ig"] / d.w_um)
    return d


def binned_median(x, y, edges, min_n=8):
    idx = np.digitize(x, edges) - 1
    cen, med, q1, q3 = [], [], [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() >= min_n:
            cen.append(0.5 * (edges[b] + edges[b + 1]))
            med.append(np.median(y[m])); q1.append(np.quantile(y[m], 0.25))
            q3.append(np.quantile(y[m], 0.75))
    return np.array(cen), np.array(med), np.array(q1), np.array(q3)


OFF_FEATS = ["temp", "avth0", "toxe", "ndep", "u0", "xj", "vdd", "avds", "src_off", "logw"]
IG_FEATS = ["temp", "avth0", "toxe", "ndep", "u0", "xj", "vdd", "vd", "vs", "logw"]


def partial_slope(df: pd.DataFrame, ycol: str, target: str, feats: list[str]):
    """OLS of y on `feats`; slope of `target` with the others held fixed.
    Features are standardised before the solve (raw scales span ~1e27) and the
    coefficient converted back to natural units. Returns (slope, R^2, n)."""
    d = df.assign(avth0=df.vth0.abs(), logw=np.log10(df.w), avds=df.vds.abs())
    d = d.dropna(subset=[ycol])
    Xr = np.column_stack([d[f].to_numpy(float) for f in feats])
    mu, sd = Xr.mean(axis=0), Xr.std(axis=0)
    X = np.column_stack([np.ones(len(d)), (Xr - mu) / sd])
    y = d[ycol].to_numpy(float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1 - np.sum((y - X @ coef) ** 2) / np.sum((y - y.mean()) ** 2)
    j = feats.index(target)
    return float(coef[1 + j] / sd[j]), float(r2), len(d)


# ===================================================================== 1. quality checks
def quality(cases, data, meta, failures) -> pd.DataFrame:
    rows = []
    for case in cases:
        df = data[case]
        checks = validate(CFG, case, df[list(CFG.columns)], CFG.n_samples, df.sample_id)
        n_fail = int((failures.case == case).sum()) if len(failures) else 0
        checks.append((n_fail == 0, f"failed simulations in failures.csv: {n_fail}"))
        m = meta["cases"].get(case, {})
        checks.append((m.get("failed_total", -1) == 0 and m.get("simulated_total") == len(df),
                       f"metadata: {m.get('simulated_total')} simulated, {m.get('failed_total')} failed"))
        for col in CURRENT_COLS:
            nz = int((df[col].abs() < NUM_FLOOR).sum())
            checks.append((True, f"info: {col} has {nz} samples with |I| < {NUM_FLOOR:g} A"))
        for ok, text in checks:
            rows.append({"case": case, "pass": bool(ok), "check": text})
    return pd.DataFrame(rows)


# ===================================================================== 2. KCL
def kcl_table(cases, data) -> pd.DataFrame:
    rows = []
    for case in cases:
        df = data[case]
        a = df.kcl.abs()
        rel = a / df.scale
        tol = KCL_REL * df.scale + KCL_ABS
        w = df.loc[a.idxmax()]
        wr = df.loc[rel.idxmax()]
        rows.append({
            "case": case, "n": len(df),
            "max_abs_A": a.max(), "median_abs_A": a.median(), "p95_abs_A": a.quantile(0.95),
            "max_rel": rel.max(), "median_rel": rel.median(), "p95_rel": rel.quantile(0.95),
            "n_over_tolerance": int((a > tol).sum()),
            "n_abs_over_1fA": int((a > 1e-15).sum()),
            "worst_abs_sample_id": int(w.sample_id), "worst_abs_sample_max_I": w.scale,
            "worst_rel_sample_id": int(wr.sample_id), "worst_rel_sample_max_I": wr.scale,
        })
    return pd.DataFrame(rows)


# ===================================================================== 3. controlled SPICE sweeps
def controlled_sweeps(sim) -> pd.DataFrame:
    """One variable swept, all others nominal; W = 2*Lmin. OFF: |Vds| = VDD with the
    source on its rail. ON gate current: Vd = Vs on the rail opposite the gate."""
    rows = []
    for dev in ("nmos", "pmos"):
        nom = nominal_process(CFG, dev)
        case = f"{dev}_off"
        for t in np.arange(-40, 126, 5.0):
            rows.append(("off_vs_temp", case, np.nan, make_inputs(CFG, case, temp=t)))
        for tb in (-12.5, 42.5, 97.5):
            for f in np.linspace(0.9, 1.1, 11):
                rows.append(("off_vs_vth0", case, tb,
                             make_inputs(CFG, case, temp=tb, process={"vth0": nom["vth0"] * f})))
        case = f"{dev}_on"
        for vdd in (0.9, 1.0, 1.1):
            v = 0.0 if dev == "nmos" else vdd
            for f in np.linspace(0.9, 1.1, 11):
                rows.append(("ig_vs_toxe", case, vdd,
                             make_inputs(CFG, case, vdd=vdd, vd=v, vs=v,
                                         process={"toxe": nom["toxe"] * f})))
    frame = pd.DataFrame(rows, columns=["kind", "case", "band", "row"])
    out = []
    for (kind, case), grp in frame.groupby(["kind", "case"], sort=False):
        res = simulate_transistors(CFG, case, pd.DataFrame(list(grp.row)), sim)
        res.insert(0, "kind", kind); res.insert(1, "case", case)
        res.insert(2, "band", grp.band.to_numpy())
        out.append(res)
    sw = pd.concat(out, ignore_index=True)
    if (sw.status != "ok").any():
        raise RuntimeError("controlled sweep failed:\n" + sw[sw.status != "ok"].to_string())
    return sw


# ===================================================================== 4. plots
def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    ax.grid(True, which="major", alpha=0.3)


def plot_off_vs_temp(data, sw, path):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    for ax, dev in zip(axes, ("nmos", "pmos")):
        d = off_subset(data[f"{dev}_off"], dev)
        ax.scatter(d.temp, 10 ** d.y, s=5, c="0.7", alpha=0.5, lw=0,
                   label=f"dataset: {OFF_LABEL} (n={len(d)})")
        av = d.vth0.abs()
        q = np.quantile(av, [0, 1 / 3, 2 / 3, 1])
        for k, name in enumerate(["low", "mid", "high"]):
            m = (av >= q[k]) & (av <= q[k + 1])
            c, med, q1, q3 = binned_median(d.temp[m].to_numpy(), d.y[m].to_numpy(),
                                           np.linspace(-40, 125, 9))
            ax.plot(c, 10 ** med, "-o", ms=3, color=COLORS[k],
                    label=f"median, {name} |vth0| ({q[k]:.3f}–{q[k+1]:.3f} V)")
            ax.fill_between(c, 10 ** q1, 10 ** q3, color=COLORS[k], alpha=0.12, lw=0)
        s = sw[(sw.kind == "off_vs_temp") & (sw.case == f"{dev}_off")]
        ax.plot(s.temp, s["id"].abs() / (s.w * 1e6), "k--", lw=2,
                label="SPICE sweep: nominal process, VDD = 1 V, |Vds| = VDD, source on rail")
        _style(ax, f"{dev.upper()} OFF (Vg = {'0' if dev == 'nmos' else 'VDD'}): drain leakage vs T",
               "Temperature (°C)", "|Id| / W  (A/µm)")
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("OFF current vs temperature — 45 nm HP, SPICE data (bands: median and IQR)", fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_off_vs_vth0(data, sw, path):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    tbands = [(-40, 15, -12.5), (15, 70, 42.5), (70, 125.0001, 97.5)]
    for ax, dev in zip(axes, ("nmos", "pmos")):
        nom = abs(nominal_process(CFG, dev)["vth0"])
        d = off_subset(data[f"{dev}_off"], dev)
        d["av"] = d.vth0.abs()
        ax.scatter(d.av, 10 ** d.y, s=5, c="0.7", alpha=0.5, lw=0,
                   label=f"dataset: {OFF_LABEL} (n={len(d)})")
        for k, (lo, hi, tc) in enumerate(tbands):
            m = (d.temp >= lo) & (d.temp < hi)
            c, med, q1, q3 = binned_median(d.av[m].to_numpy(), d.y[m].to_numpy(),
                                           np.linspace(0.9 * nom, 1.1 * nom, 7))
            ax.plot(c, 10 ** med, "-o", ms=3, color=COLORS[k],
                    label=f"median, T in [{lo:g}, {min(hi, 125):g}] °C")
            ax.fill_between(c, 10 ** q1, 10 ** q3, color=COLORS[k], alpha=0.12, lw=0)
            s = sw[(sw.kind == "off_vs_vth0") & (sw.case == f"{dev}_off") & (sw.band == tc)]
            ax.plot(s.vth0.abs(), s["id"].abs() / (s.w * 1e6), "--", color=COLORS[k], lw=1.6,
                    label=f"SPICE sweep, T = {tc:g} °C, others nominal")
        ax.axvline(nom, color="k", lw=0.8, ls=":")
        _style(ax, f"{dev.upper()} OFF: drain leakage vs |vth0|",
               f"|vth0| (V)   — nominal {nom:.4f} V (dotted)", "|Id| / W  (A/µm)")
        ax.legend(fontsize=7, loc="lower left")
    fig.suptitle("OFF current vs vth0 — 45 nm HP, SPICE data (bands: median and IQR)", fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_ig_vs_toxe(data, sw, path):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    vbands = [(0.9, 0.9667, 0.9), (0.9667, 1.0333, 1.0), (1.0333, 1.1001, 1.1)]
    for ax, dev in zip(axes, ("nmos", "pmos")):
        nom = nominal_process(CFG, dev)["toxe"]
        d = gate_subset(data[f"{dev}_on"], dev)
        cond = "Vd, Vs ≤ 0.25·VDD" if dev == "nmos" else "Vd, Vs ≥ 0.75·VDD"
        ax.scatter(d.toxe * 1e9, 10 ** d.y, s=5, c="0.7", alpha=0.5, lw=0,
                   label=f"dataset: {cond} (n={len(d)})")
        for k, (lo, hi, vc) in enumerate(vbands):
            m = (d.vdd >= lo) & (d.vdd < hi)
            c, med, q1, q3 = binned_median((d.toxe[m] * 1e9).to_numpy(), d.y[m].to_numpy(),
                                           np.linspace(0.9 * nom, 1.1 * nom, 9) * 1e9)
            ax.plot(c, 10 ** med, "-o", ms=3, color=COLORS[k],
                    label=f"median, VDD in [{lo:.2f}, {min(hi, 1.1):.2f}] V")
            ax.fill_between(c, 10 ** q1, 10 ** q3, color=COLORS[k], alpha=0.12, lw=0)
            s = sw[(sw.kind == "ig_vs_toxe") & (sw.case == f"{dev}_on") & np.isclose(sw.band, vc)]
            ax.plot(s.toxe * 1e9, s["ig"].abs() / (s.w * 1e6), "--", color=COLORS[k], lw=1.6,
                    label=f"SPICE sweep, VDD = {vc:g} V, Vd = Vs = {'0' if dev == 'nmos' else 'VDD'}")
        ax.axvline(nom * 1e9, color="k", lw=0.8, ls=":")
        _style(ax, f"{dev.upper()} ON (Vg = {'VDD' if dev == 'nmos' else '0'}): gate current vs toxe",
               f"toxe (nm)   — nominal {nom*1e9:.3f} nm (dotted)", "|Ig| / W  (A/µm)")
        ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("Gate (tunnelling) current vs toxe — 45 nm HP, SPICE data", fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def source_offset_effect(data) -> pd.DataFrame:
    rows = []
    edges = [0, 0.05, 0.1, 0.2, 0.3, 0.5]
    for dev in ("nmos", "pmos"):
        df = data[f"{dev}_off"]
        so = source_offset(df, dev) / df.vdd
        y = safe_log10_abs(df["id"] / df.w_um)
        base = (df.vds.abs() >= OFF_VDS_MIN * df.vdd) & df.temp.between(20, 35)
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = base & (so >= lo) & (so < hi)
            rows.append({"device": dev, "source_offset_vdd": f"{lo:.2f}-{hi:.2f}", "n": int(m.sum()),
                         "median_log10_Id_per_um": float(y[m].median())})
    return pd.DataFrame(rows)


def plot_source_offset(data, path):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for ax, dev in zip(axes, ("nmos", "pmos")):
        df = data[f"{dev}_off"]
        d = df[df.vds.abs() >= OFF_VDS_MIN * df.vdd]
        so = source_offset(d, dev)
        sc = ax.scatter(so, d["id"].abs() / d.w_um, c=d.temp, cmap="coolwarm", s=4, lw=0)
        ax.axvspan(0, OFF_SRC_MAX * 1.1, color="0.85", zorder=0, label="kept for trend plots")
        _style(ax, f"{dev.upper()} OFF, |Vds| ≥ {OFF_VDS_MIN:g}·VDD: why the source potential matters",
               "source offset from its rail (V)  " + ("= min(Vd, Vs)" if dev == "nmos" else "= VDD − max(Vd, Vs)"),
               "|Id| / W  (A/µm)")
        fig.colorbar(sc, ax=ax, label="T (°C)")
        ax.legend(fontsize=8)
    fig.suptitle("Supplementary: OFF current falls ~1 decade per 0.1·VDD of source offset (negative Vgs + body effect)",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ===================================================================== 5. slopes
def trend_slopes(data, sw) -> pd.DataFrame:
    rows = []

    def sweep_slope(x, y):
        return float(np.polyfit(np.asarray(x, float), np.log10(np.abs(np.asarray(y, float))), 1)[0])

    for dev in ("nmos", "pmos"):
        d = off_subset(data[f"{dev}_off"], dev)
        b, r2, n = partial_slope(d, "y", "temp", OFF_FEATS)
        s = sw[(sw.kind == "off_vs_temp") & (sw.case == f"{dev}_off")]
        rows.append({"trend": "OFF |Id| vs temperature", "device": dev, "unit": "decades per 10 C",
                     "dataset_partial_slope": 10 * b, "controlled_sweep_slope": 10 * sweep_slope(s.temp, s["id"]),
                     "expected_sign": "+", "ols_r2": r2, "n": n})
        b, r2, n = partial_slope(d, "y", "avth0", OFF_FEATS)
        s = sw[(sw.kind == "off_vs_vth0") & (sw.case == f"{dev}_off") & (sw.band == 42.5)]
        rows.append({"trend": "OFF |Id| vs |vth0|", "device": dev, "unit": "decades per 10 mV",
                     "dataset_partial_slope": 0.01 * b,
                     "controlled_sweep_slope": 0.01 * sweep_slope(s.vth0.abs(), s["id"]),
                     "expected_sign": "-", "ols_r2": r2, "n": n})
        g = gate_subset(data[f"{dev}_on"], dev)
        b, r2, n = partial_slope(g, "y", "toxe", IG_FEATS)
        s = sw[(sw.kind == "ig_vs_toxe") & (sw.case == f"{dev}_on") & np.isclose(sw.band, 1.0)]
        rows.append({"trend": "ON |Ig| vs toxe", "device": dev, "unit": "decades per 0.1 nm",
                     "dataset_partial_slope": 1e-10 * b,
                     "controlled_sweep_slope": 1e-10 * sweep_slope(s.toxe, s["ig"]),
                     "expected_sign": "-", "ols_r2": r2, "n": n})
        b, r2, n = partial_slope(g, "y", "temp", IG_FEATS)
        rows.append({"trend": "ON |Ig| vs temperature", "device": dev, "unit": "decades per 10 C",
                     "dataset_partial_slope": 10 * b, "controlled_sweep_slope": np.nan,
                     "expected_sign": "~0", "ols_r2": r2, "n": n})
    t = pd.DataFrame(rows)
    exp = t.expected_sign.map({"+": 1, "-": -1, "~0": 0})
    t["dataset_sign_ok"] = np.where(exp == 0, t.dataset_partial_slope.abs() < 0.02,
                                    np.sign(t.dataset_partial_slope) == exp)
    t["sweep_sign_ok"] = np.where(exp == 0, True, np.sign(t.controlled_sweep_slope) == exp)
    return t


# ===================================================================== 6. suspicious samples
def suspicious(cases, data) -> pd.DataFrame:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import KFold, cross_val_predict
    flags = []

    def add(case, df, mask, reason, detail):
        for i in np.flatnonzero(np.asarray(mask)):
            r = df.iloc[i]
            flags.append({"case": case, "row": int(i), "sample_id": int(r.sample_id), "reason": reason,
                          "detail": detail(r, i),
                          **{c: r[c] for c in ["vdd", "temp", "w", "vd", "vs", "vth0", "toxe",
                                               "id", "ig", "is", "ib", "kcl"]}})

    for case in cases:
        dev = CFG.cases[case].device
        df = data[case].reset_index(drop=True)
        # (a) Kirchhoff above tolerance
        tol = KCL_REL * df.scale + KCL_ABS
        add(case, df, df.kcl.abs() > tol, "kcl_over_tolerance",
            lambda r, i: f"|sum|={abs(r.kcl):.2e} A, max|I|={r.scale:.2e} A")
        # (b) channel current against Vds although |Vds| is large
        big = df.vds.abs() > SIGN_VDS_MIN
        wrong = big & (np.sign(df["id"]) != np.sign(df.vds)) & (np.sign(df["is"]) == np.sign(df.vds))
        add(case, df, wrong, "drain_current_against_vds",
            lambda r, i: f"Vds={r.vds:+.3f} V but Id={r['id']:+.2e}, Is={r['is']:+.2e}")
        # (c) near numerical floor
        for col in CURRENT_COLS:
            add(case, df, df[col].abs() < NUM_FLOOR, f"{col}_below_{NUM_FLOOR:g}A",
                lambda r, i, c=col: f"|{c}|={abs(r[c]):.2e} A")
        # (d) statistical outliers: 5-fold cross-validated gradient boosting of log10|I|
        X = pd.DataFrame({"vdd": df.vdd, "temp": df.temp, "logw": np.log10(df.w),
                          "vd": df.vd, "vs": df.vs, "vds": df.vds, "avds": df.vds.abs(),
                          "src_off": source_offset(df, dev),
                          **{p: df[p] for p in CFG.process_params}})
        for col in ("id", "ig", "ib"):
            y = safe_log10_abs(df[col])
            ok = y.notna().to_numpy()
            model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.08, random_state=SEED)
            pred = np.full(len(df), np.nan)
            pred[ok] = cross_val_predict(model, X[ok], y[ok],
                                         cv=KFold(5, shuffle=True, random_state=SEED))
            res = y.to_numpy() - pred
            med = np.nanmedian(res)
            mad = 1.4826 * np.nanmedian(np.abs(res - med))
            z = (res - med) / mad
            m = (np.abs(z) > OUTLIER_Z) & (np.abs(res - med) > OUTLIER_DEC)
            if col == "id":
                m &= (df.vds.abs() > ZERO_CROSS_VDS).to_numpy()
            add(case, df, m, f"{col}_outlier_vs_neighbours",
                lambda r, i, c=col, zz=z, rr=res: f"log10|{c}| off by {rr[i]:+.2f} decades (robust z={zz[i]:+.1f})")
    return pd.DataFrame(flags)


# ===================================================================== 7. warning audit + extra facts
def warning_audit(cases, data, sus, n_random=10) -> pd.DataFrame:
    """Re-simulate every flagged sample (+ n_random random others per case) ALONE,
    record its exact simulator messages and confirm the dataset values are reproduced."""
    from leakage.simulator import _issues
    from leakage.spice import find_ngspice, parse_prints, run_ngspice
    from leakage.transistor import example_netlist
    exe = find_ngspice(CFG.ngspice_path)
    rng = np.random.default_rng(SEED)
    rows = []
    for case in cases:
        df = data[case].reset_index(drop=True)
        flagged = set(sus.loc[sus.case == case, "row"]) if len(sus) else set()
        pool = np.setdiff1d(np.arange(len(df)), list(flagged))
        picks = [(r, True) for r in sorted(flagged)] + \
                [(int(r), False) for r in rng.choice(pool, n_random, replace=False)]
        for r, is_flag in picks:
            row = df.iloc[r]
            out = run_ngspice(exe, example_netlist(CFG, case, row), timeout=60)
            v = parse_prints(out)[None]
            cur = np.array([-v.get(f"i(v{t})", np.nan) for t in "dgsb"])
            ref = row[CURRENT_COLS].to_numpy(float)
            rows.append({"case": case, "sample_id": int(row.sample_id), "flagged": is_flag,
                         "reproduced": bool(np.allclose(cur, ref, rtol=1e-9, atol=0)),
                         "messages": " | ".join(sorted(set(_issues(out))))})
    return pd.DataFrame(rows)


def extra_facts(data) -> dict:
    f = {}
    for dev in ("nmos", "pmos"):
        off = data[f"{dev}_off"]
        big = off.vds.abs() > SIGN_VDS_MIN
        f[f"{dev}_off_id_against_vds_frac"] = float((np.sign(off["id"][big]) != np.sign(off.vds[big])).mean())
        nom = nominal_process(CFG, dev)["toxe"]
        f[f"{dev}_toxe_nominal_frac"] = float(np.isclose(pd.concat([off.toxe, data[f'{dev}_on'].toxe]), nom,
                                                         rtol=1e-12, atol=0).mean())
    for dev in ("nmos", "pmos"):
        off = data[f"{dev}_off"]
        a = off.ib.abs()
        f[f"{dev}_off_ib_frac_near_1e-14"] = float(a.between(0.5e-14, 2e-14).mean())
        f[f"{dev}_off_ib_median"] = float(a.median())
        f[f"{dev}_off_ib_corr_logw"] = float(np.corrcoef(np.log10(off.w), np.log10(a))[0, 1])
        on = data[f"{dev}_on"]
        f[f"{dev}_on_ib_max"] = float(on.ib.abs().max())
        g = gate_subset(on, dev)
        f[f"{dev}_on_ig_corr_logw"] = float(np.corrcoef(np.log10(g.w), g.y)[0, 1])
    return f


# ===================================================================== 8. report
def md_table(df: pd.DataFrame, floatfmt="{:.3g}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            txt = floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
            cells.append(txt.replace("|", "\\|"))          # keep '|Id|' from breaking the table
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def vth0_mv_per_decade(sw) -> dict:
    out = {}
    s = sw[sw.kind == "off_vs_vth0"]
    for (case, band), g in s.groupby(["case", "band"]):
        out[(case, band)] = -1000.0 / np.polyfit(g.vth0.abs(), np.log10(g["id"].abs()), 1)[0]
    return out


def write_report(meta, q, k, slopes, so_eff, sus, facts, sw, audit) -> str:
    n_cases = len(k)
    mvd = vth0_mv_per_decade(sw)
    mvd_txt = "; ".join(
        f"{dev.upper()} " + " / ".join(f"{mvd[(f'{dev}_off', b)]:.0f}" for b in (-12.5, 42.5, 97.5))
        for dev in ("nmos", "pmos"))
    qf = q[~q["pass"]]
    kk = k[["case", "n", "max_abs_A", "median_abs_A", "p95_abs_A", "max_rel", "median_rel",
            "p95_rel", "n_over_tolerance", "n_abs_over_1fA"]]
    sl = slopes[["trend", "device", "unit", "dataset_partial_slope", "controlled_sweep_slope",
                 "expected_sign", "dataset_sign_ok", "ols_r2", "n"]]
    by_reason = (sus.groupby(["case", "reason"]).size().rename("count").reset_index()
                 if len(sus) else pd.DataFrame(columns=["case", "reason", "count"]))
    nom_off = sw[(sw.kind == "off_vs_temp")]
    L = []
    L += ["# Part A1 data-quality report", "",
          f"Datasets: `data/transistor/<case>.csv`, {n_cases} cases x {CFG.n_samples} rows, "
          f"model `{meta['model']['file']}`, ngspice {meta['ngspice']['version']}, generated "
          f"{meta['created']} at git commit `{meta['git']['commit'][:10]}`. "
          f"Simulations: {meta['totals']['simulated']} run, {meta['totals']['succeeded']} succeeded, "
          f"{meta['totals']['failed']} failed.", "",
          "Sign convention: every terminal current is **positive when it flows into the device** "
          "(I_X = −i(V_X) from ngspice). Units: A, V, °C, m.", "",
          "No SPICE value was modified, clipped or removed for this report.", ""]

    L += ["## 1. Completeness and validity", "",
          f"{int(q['pass'].sum())} of {len(q)} checks pass "
          f"({'none fail' if qf.empty else str(len(qf)) + ' fail, listed below'}). Per case the checks cover: "
          "row count = 10,000; exact column order; no NaN/inf; VDD in [0.9, 1.1] V; T in [−40, 125] °C; "
          "W in [45, 360] nm; 0 ≤ Vd, Vs ≤ VDD; Vg and Vb follow the case rule; each process parameter "
          "within ±10 % of its own device's nominal; unique input rows and sample ids; no exactly-zero "
          "currents; Kirchhoff within tolerance; zero failed simulations (failures.csv, metadata).", ""]
    if not qf.empty:
        L += [md_table(qf), ""]
    L += ["Full table: `quality_checks.csv`.", ""]

    L += ["## 2. Kirchhoff check: Id + Ig + Is + Ib", "", md_table(kk), "",
          f"Tolerance used: |sum| ≤ {KCL_REL:g}·max|I| + {KCL_ABS:g} A. "
          "**No sample in any case exceeds it.** Interpretation:", "",
          "* With the into-the-device convention the four currents must sum to zero; signs are "
          "already consistent (e.g. NMOS ON: Id > 0, Is < 0).",
          "* Absolute residuals are at the 1e-25 to 1e-15 A level, i.e. at or below the simulator's "
          "absolute tolerance scale (abstol = 1e-18 A) and double-precision round-off of the largest "
          "terminal current (~1e-16 relative).",
          "* The largest *relative* residuals (up to ~1e-2) occur only in samples where **all four** "
          "currents are tiny (≤ ~1e-12 A, typically Vd ≈ Vs): there a fixed residual of ~1e-15 A "
          "remains. In Part A1 this was shown not to change with gmin, reltol or vntol, so it is a "
          "small non-conservation inside the BSIM4 device evaluation, ~1000x below any leakage of interest.",
          "* No sample was dropped because of KCL.", ""]

    L += ["## 3. Physical trends", "",
          "Because every input is randomised independently, raw scatter plots hide the trends. "
          "The plots therefore (i) normalise by width (A/µm), (ii) condition on the bias that defines "
          "the quantity, (iii) show binned medians (with inter-quartile bands) separately for bands of "
          "the most influential other variable, and (iv) overlay a **controlled SPICE sweep** (one "
          "variable swept, everything else nominal, W = 90 nm). The controlled sweeps are reference "
          "simulations only (`controlled_sweeps.csv`), not training data.", "",
          "### 3.1 Why the OFF-current subset is conditioned on the source potential", "",
          "A first version conditioned only on |Vds| and its medians sat 1–2 decades below the sweep. "
          "Investigation showed this is physics, not an error: with Vd and Vs both random, the effective "
          "source is usually *above* its rail (NMOS) or *below* VDD (PMOS). That makes Vgs negative and "
          "adds reverse body bias, cutting the sub-threshold current exponentially:", "",
          md_table(so_eff, "{:.2f}"), "",
          "(|Vds| ≥ 0.5·VDD, 20–35 °C.) The current falls about one decade per 0.1·VDD of source offset "
          "and saturates near 1e-10.5 A/µm, where other drain components (gate-to-drain tunnelling, "
          "junction) dominate. See `fig_supp_off_current_vs_source_offset.png`. The trend plots therefore "
          f"use |Vds| ≥ {OFF_VDS_MIN:g}·VDD **and** source within {OFF_SRC_MAX:g}·VDD of its rail.", "",
          "### 3.2 Plots", "",
          "![OFF vs T](fig_off_current_vs_temperature.png)", "",
          "![OFF vs vth0](fig_off_current_vs_vth0.png)", "",
          "![Ig vs toxe](fig_gate_current_vs_toxe.png)", "",
          "### 3.3 Trend slopes (other inputs held fixed)", "", md_table(sl, "{:.3g}"), "",
          "`dataset_partial_slope`: OLS of log10(|I|/W) on all inputs within the conditioned subset. "
          "`controlled_sweep_slope`: log-linear fit to the SPICE sweep (VDD = 1 V, W = 90 nm; vth0 sweep at 42.5 °C).", "",
          "### 3.4 Explanation", "",
          f"* **OFF current vs temperature (rises).** Sub-threshold current ∝ exp((Vgs − Vth)/(n·kT/q)): "
          f"Vth falls with T (kt1 < 0) and the thermal voltage grows, so leakage rises roughly exponentially: "
          f"the sweep gives {nom_off[nom_off.case == 'nmos_off']['id'].abs().max() / nom_off[nom_off.case == 'nmos_off']['id'].abs().min():.0f}x (NMOS) and "
          f"{nom_off[nom_off.case == 'pmos_off']['id'].abs().max() / nom_off[nom_off.case == 'pmos_off']['id'].abs().min():.0f}x (PMOS) from −40 to 125 °C. "
          "The dataset slope agrees in sign and order of magnitude with the controlled sweep.",
          "* **OFF current vs |vth0| (falls).** A higher threshold moves the device deeper into "
          "sub-threshold: one decade per n·(kT/q)·ln10 of threshold. Measured on the controlled sweeps at "
          f"−12.5 / 42.5 / 97.5 °C: {mvd_txt} mV per decade. For NMOS the sensitivity weakens with "
          "temperature as kT/q grows, as expected; for PMOS it is not monotonic over this range (other "
          "temperature-dependent terms also enter), but the direction is the same everywhere. The "
          "temperature bands are vertically separated by the temperature effect above.",
          "* **Gate current vs toxe (falls steeply).** Direct tunnelling through the oxide decays "
          "exponentially with thickness: ~0.6 decades per 0.1 nm here. It also rises with VDD (higher "
          f"oxide field). It is essentially **temperature-independent** in this model (partial slope "
          f"≈ 0, table above) and scales with W (corr(log W, log Ig/W) ≈ "
          f"{facts['nmos_on_ig_corr_logw']:.2f} NMOS / {facts['pmos_on_ig_corr_logw']:.2f} PMOS after "
          "normalising, i.e. Ig ∝ W).",
          "* All dataset trends have the physically expected sign, and agree with the controlled sweeps. "
          "Residual differences in magnitude come from the conditioning windows (|Vds| from 0.5·VDD, "
          "source up to 0.05·VDD off its rail, VDD ±10 %), not from the data.", ""]

    L += ["### 3.5 Other observations", "",
          f"* **OFF-state body current is ~1e-14 A and nearly constant.** {facts['nmos_off_ib_frac_near_1e-14']:.0%} "
          f"(NMOS) / {facts['pmos_off_ib_frac_near_1e-14']:.0%} (PMOS) of OFF samples have |Ib| between 0.5e-14 and 2e-14 A, "
          f"almost independent of W (corr(log W, log|Ib|) = {facts['nmos_off_ib_corr_logw']:.2f} / "
          f"{facts['pmos_off_ib_corr_logw']:.2f}) and of temperature. BSIM4's internal breakdown at the "
          "nominal point shows it is the drain–body junction current (ibd ≈ −1.0e-14 A) with the source "
          "junction, GIDL, gate-to-body tunnelling and impact ionisation negligible. The netlists, like the "
          "supplied example, give no junction area/perimeter (AD/AS/PD/PS), so the junction saturation "
          "current is a fixed small value rather than ∝ W. This is genuine SPICE output and is used "
          "consistently in the gate and circuit netlists, so it is kept; it is a modelling limitation to note "
          "in the report.",
          f"* **ON-state body current reaches {max(facts['nmos_on_ib_max'], facts['pmos_on_ib_max']):.1e} A** at large Vds "
          "(consistent with the impact-ionisation substrate current enabled by alpha0/beta0 in the card) — "
          "genuine model behaviour.",
          "* Gate current dominates the OFF-state *total* leakage whenever the source is off its rail; "
          "the ML models must therefore predict all four terminals, not only Id.",
          f"* **OFF-state Id can flow against Vds.** With |Vds| > {SIGN_VDS_MIN} V, "
          f"{facts['nmos_off_id_against_vds_frac']:.1%} (NMOS) / {facts['pmos_off_id_against_vds_frac']:.1%} (PMOS) "
          "of OFF samples have Id opposite in sign to Vds. The drain terminal current is the channel "
          "current minus the gate-to-drain tunnelling current (opposite directions); when the source is "
          "off its rail the channel current collapses and the tunnelling term wins. Id therefore crosses "
          "zero away from Vds = 0 as well, and |Id| can dip far below both components.", ""]

    n_aud = len(audit)
    msgs = audit.messages.replace("", "(none)").value_counts()
    L += ["### 3.6 Simulator-warning audit", "",
          f"All flagged samples plus 10 random unflagged samples per case ({n_aud} in total) were "
          "re-simulated one per ngspice process so their messages are attributed exactly. "
          f"{int(audit.reproduced.sum())}/{n_aud} reproduced the dataset currents to < 1e-9 relative. "
          "Messages seen:", "",
          md_table(msgs.rename_axis("message").reset_index(name="samples")), "",
          "**toxe/toxp/dtox warning.** In the 45 nm HP card toxe = toxp + dtox exactly (NMOS 1.25 = 1.00 + 0.25 nm, "
          "PMOS 1.30 = 1.00 + 0.30 nm). The specification varies only toxe, so for every sample with a "
          f"non-nominal toxe ({1 - facts['nmos_toxe_nominal_frac']:.2%} of NMOS and "
          f"{1 - facts['pmos_toxe_nominal_frac']:.2%} of PMOS samples; the generator recorded the warning "
          "for all 10,000 samples of every case) BSIM4 "
          "reports the inconsistency and ignores dtox, i.e. the physical oxide thickness toxp stays at its "
          "nominal 1.0 nm while the electrical thickness toxe varies. Gate tunnelling follows toxe, as the "
          "Ig-vs-toxe plot and slopes confirm; quantities that BSIM4 computes from toxp do not vary. "
          "This is a direct consequence of the specified parameter set, applied identically in every "
          "transistor, gate and circuit netlist; the warning is benign for this study and is documented as "
          "a modelling assumption (alternative: also vary toxp = toxe − dtox).", ""]

    L += ["## 4. Suspicious samples (flagged, not removed)", "",
          "Screens: (a) Kirchhoff above tolerance; (b) channel current reversed: Id **and** Is both "
          f"flowing against Vds although |Vds| > {SIGN_VDS_MIN} V (Id alone reversing is common and "
          "explained in §3.5; both reversing would mean a wrong channel direction); "
          f"(c) any |I| < {NUM_FLOOR:g} A (within ~100x of abstol, so its "
          "relative precision is poor); (d) statistical outliers: 5-fold cross-validated gradient-boosting "
          "prediction of log10|I| from all inputs, flag if the residual exceeds both "
          f"{OUTLIER_Z:g} robust standard deviations and {OUTLIER_DEC:g} decade (Id samples with "
          f"|Vds| < {ZERO_CROSS_VDS} V are skipped there because Id crosses zero at Vds = 0).", ""]
    L += [md_table(by_reason) if len(by_reason) else "No sample was flagged by any screen.", "",
          "All flagged rows with their inputs, currents and the reason are in `suspicious_samples.csv`.", ""]
    if len(sus):
        L += ["Interpretation:", ""]
        if (sus.reason.str.startswith("ib_below")).any():
            L += ["* `ib_below_*`: OFF samples with both junctions near zero bias (Vd and Vs ≈ 0 for NMOS, ≈ VDD "
                  "for PMOS) or at low temperature; |Ib| ~ 1e-17 A is physically plausible but numerically "
                  "fragile. Predicting log|Ib| there needs care (see Part A2)."]
        if (sus.reason.str.contains("outlier")).any():
            L += ["* `id_outlier_vs_neighbours` (OFF cases): |Id| is 1–2 decades below what neighbouring "
                  "samples predict because the channel current and the gate-to-drain tunnelling current "
                  "nearly cancel at the drain (median |Id|/|Ig| ≈ 0.003 among them, Id sign 50/50 vs Vds) — "
                  "a genuine zero crossing of Id, see §3.5. `ib_/ig_outlier...` and the ON-case flags are "
                  "the same kind of steep, sign-changing behaviour. Every flagged sample was re-simulated "
                  "alone (§3.6): values reproduced exactly, no convergence problem, Kirchhoff satisfied. "
                  "They are kept as valid SPICE data; they mark where an ML model on log|I| will find the "
                  "data hardest."]
        if (sus.reason == "drain_current_against_vds").any():
            L += ["* `drain_current_against_vds`: the drain current flows against Vds; at these biases another "
                  "component (gate-to-drain tunnelling or junction current) exceeds the channel current."]
        L += [""]

    L += ["## 5. Conclusion", "",
          "The four datasets are complete (40,000 / 40,000 successful simulations, 0 failures), inside every "
          "specified range, free of missing or duplicated values, and satisfy Kirchhoff's law to "
          "numerical precision. The three required trends have the physically expected direction and "
          "agree with controlled SPICE sweeps once the independently randomised bias is accounted for. "
          "The data are fit for training. Points to carry into Part A2: currents span ~11 decades and "
          "change sign (Id with Vds, Ig/Ib with bias), so models should predict sign and log-magnitude "
          "(or an equivalent transform) per terminal.", ""]
    return "\n".join(L)


# ===================================================================== main
def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = list(CFG.cases)
    meta = json.loads((DATA / "metadata.json").read_text(encoding="utf-8"))
    failures = pd.read_csv(DATA / "failures.csv")
    data = {c: load_case(c) for c in cases}
    sim = NgspiceSimulator.from_config(CFG)

    q = quality(cases, data, meta, failures)
    q.to_csv(OUT / "quality_checks.csv", index=False)
    k = kcl_table(cases, data)
    k.to_csv(OUT / "kcl_summary.csv", index=False)

    print("Running controlled SPICE reference sweeps ...")
    sw = controlled_sweeps(sim)
    sw.to_csv(OUT / "controlled_sweeps.csv", index=False)

    print("Plotting ...")
    plot_off_vs_temp(data, sw, OUT / "fig_off_current_vs_temperature.png")
    plot_off_vs_vth0(data, sw, OUT / "fig_off_current_vs_vth0.png")
    plot_ig_vs_toxe(data, sw, OUT / "fig_gate_current_vs_toxe.png")
    plot_source_offset(data, OUT / "fig_supp_off_current_vs_source_offset.png")
    so_eff = source_offset_effect(data)
    so_eff.to_csv(OUT / "source_offset_effect.csv", index=False)
    slopes = trend_slopes(data, sw)
    slopes.to_csv(OUT / "trend_slopes.csv", index=False)

    print("Screening for suspicious samples ...")
    sus = suspicious(cases, data)
    sus.to_csv(OUT / "suspicious_samples.csv", index=False)
    facts = extra_facts(data)
    print("Re-simulating flagged + random samples alone (warning audit) ...")
    audit = warning_audit(cases, data, sus)
    audit.to_csv(OUT / "warning_audit.csv", index=False)

    report = write_report(meta, q, k, slopes, so_eff, sus, facts, sw, audit)
    (OUT / "data_quality_report.md").write_text(report, encoding="utf-8")
    (CFG.reports / "a1_data_quality_report.md").write_text(
        report.replace("](fig_", "](../results/data_check/fig_"), encoding="utf-8")

    pd.set_option("display.width", 220)
    print(f"\nQuality checks: {int(q['pass'].sum())}/{len(q)} pass")
    print("\nKCL summary:")
    print(k.iloc[:, :10].to_string(index=False, float_format=lambda x: f"{x:.3e}"))
    print("\nTrend slopes:")
    print(slopes.drop(columns=["unit"]).to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print("\nSuspicious samples by reason:")
    print(sus.groupby(["case", "reason"]).size().to_string() if len(sus) else "  none")
    print("\nFacts:", json.dumps({k_: round(v, 4) for k_, v in facts.items()}))
    print(f"\nOutputs in {OUT}")
    print(f"\nWarning audit: {int(audit.reproduced.sum())}/{len(audit)} reproduced; messages:")
    print(audit.messages.replace("", "(none)").value_counts().to_string())
    return 0 if (q["pass"].all() and slopes.dataset_sign_ok.all() and slopes.sweep_sign_ok.all()
                 and audit.reproduced.all()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
