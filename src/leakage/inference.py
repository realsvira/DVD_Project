"""Inference API for the trained transistor leakage models (Part A2 -> Parts B/C).

    from leakage.inference import LeakagePredictor
    lp = LeakagePredictor()                       # loads trained_models/<case>.joblib lazily
    out = lp.predict("nmos", vd=1.0, vg=0.0, vs=0.0, vb=0.0, vdd=1.0, temp=27.0, w=90e-9)
    out[["id", "ig", "is", "ib"]]                 # A, positive = INTO the device terminal

* device: "nmos" | "pmos".  state: "on" | "off" | None (inferred from Vg:
  NMOS is ON when Vg = VDD, PMOS is ON when Vg = 0).
* Terminal voltages, VDD, temp (deg C), w (m) and process parameters may be scalars
  or arrays (broadcast). Process parameters default to the nominal model values.
* The case rules are enforced: Vg and Vb must equal the case values (NMOS ON:
  Vg=VDD,Vb=0; NMOS OFF: Vg=0,Vb=0; PMOS ON: Vg=0,Vb=VDD; PMOS OFF: Vg=VDD,Vb=VDD).
* Inputs outside the training ranges are flagged in column `extrapolated`
  (and raise with strict=True).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, load_config
from .ml import INPUTS, PROCESS, TARGETS, CurrentModel

RULE_TOL = 1e-6      # V: allowed deviation of Vg / Vb from the case rule


class LeakagePredictor:
    def __init__(self, models_dir: Path | str | None = None, cfg: Config | None = None):
        self.cfg = cfg or load_config()
        self.models_dir = Path(models_dir) if models_dir else self.cfg.trained_models
        self._models: dict[str, CurrentModel] = {}

    # ------------------------------------------------------------------ models
    def model(self, case: str) -> CurrentModel:
        if case not in self._models:
            import joblib
            path = self.models_dir / f"{case}.joblib"
            if not path.is_file():
                raise FileNotFoundError(f"{path} not found; run scripts/a2_train_models.py first")
            m = joblib.load(path)
            if m.case != case:
                raise ValueError(f"{path} holds case {m.case}, expected {case}")
            self._models[case] = m
        return self._models[case]

    def info(self, case: str) -> dict:
        m = self.model(case)
        return {k: m.meta.get(k) for k in ("algorithm", "params", "transform", "validation_score",
                                            "created", "n_train", "dataset_sha256")}

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def infer_state(device: str, vg, vdd) -> np.ndarray:
        vg, vdd = np.asarray(vg, float), np.asarray(vdd, float)
        at_vdd = np.isclose(vg, vdd, atol=RULE_TOL, rtol=0)
        at_gnd = np.isclose(vg, 0.0, atol=RULE_TOL, rtol=0)
        if not np.all(at_vdd | at_gnd):
            raise ValueError("Vg must be exactly 0 or VDD to infer the ON/OFF state")
        on = at_vdd if device == "nmos" else at_gnd
        return np.where(on, "on", "off")

    def _check_rule(self, case: str, df: pd.DataFrame) -> None:
        c = self.cfg.cases[case]
        for col, frac in (("vg", c.vg_frac), ("vb", c.vb_frac)):
            bad = ~np.isclose(df[col], frac * df["vdd"], atol=RULE_TOL, rtol=0)
            if bad.any():
                raise ValueError(f"{case}: {col} must equal {frac:g}*VDD "
                                 f"({int(bad.sum())} rows violate it)")

    def _extrapolated(self, case: str, df: pd.DataFrame) -> np.ndarray:
        cfg, eps = self.cfg, 1e-9
        nom = self.model(case).nominal
        lo, hi = cfg.vdd_range
        out = ~df.vdd.between(lo - eps, hi + eps)
        out |= ~df.temp.between(cfg.temp_range[0] - eps, cfg.temp_range[1] + eps)
        out |= ~df.w.between(cfg.w_range[0] * (1 - eps), cfg.w_range[1] * (1 + eps))
        for v in ("vd", "vs"):
            out |= (df[v] < -1e-6) | (df[v] > df.vdd + 1e-6)
        for p in PROCESS:
            r = df[p] / nom[p]
            out |= ~r.between(1 - cfg.process_tolerance - eps, 1 + cfg.process_tolerance + eps)
        return out.to_numpy()

    # ------------------------------------------------------------------ predict
    def predict_frame(self, device: str, df: pd.DataFrame, state: str | None = None,
                      strict: bool = False) -> pd.DataFrame:
        """df needs columns vdd, temp, w, vd, vs, vg, vb (+ optional process params)."""
        device = device.lower()
        if device not in ("nmos", "pmos"):
            raise ValueError("device must be 'nmos' or 'pmos'")
        df = df.copy().reset_index(drop=True)
        states = (np.full(len(df), state.lower()) if state else
                  self.infer_state(device, df["vg"], df["vdd"]))
        out = pd.DataFrame(index=df.index, columns=["case", *TARGETS, "extrapolated"])
        for st in np.unique(states):
            if st not in ("on", "off"):
                raise ValueError("state must be 'on' or 'off'")
            case = f"{device}_{st}"
            m = self.model(case)
            rows = df[states == st].copy()
            for p in PROCESS:
                if p not in rows or rows[p].isna().any():
                    rows[p] = rows[p].fillna(m.nominal[p]) if p in rows else m.nominal[p]
            self._check_rule(case, rows)
            ext = self._extrapolated(case, rows)
            if strict and ext.any():
                raise ValueError(f"{case}: {int(ext.sum())} rows outside the training ranges")
            pred = m.predict(rows[INPUTS])
            out.loc[rows.index, TARGETS] = pred[TARGETS].to_numpy()
            out.loc[rows.index, "case"] = case
            out.loc[rows.index, "extrapolated"] = ext
        out[TARGETS] = out[TARGETS].astype(float)
        out["extrapolated"] = out["extrapolated"].astype(bool)
        return out

    def predict(self, device: str, *, vd, vg, vs, vb, vdd, temp, w, vth0=None, toxe=None,
                ndep=None, u0=None, xj=None, state: str | None = None,
                strict: bool = False) -> pd.DataFrame:
        """Scalar or array inputs (broadcast). Returns DataFrame: case, id, ig, is, ib (A), extrapolated."""
        cols = dict(vd=vd, vg=vg, vs=vs, vb=vb, vdd=vdd, temp=temp, w=w,
                    vth0=vth0, toxe=toxe, ndep=ndep, u0=u0, xj=xj)
        given = {k: np.asarray(v, float) for k, v in cols.items() if v is not None}
        shape = np.broadcast_shapes(*(v.shape for v in given.values()))
        df = pd.DataFrame({k: np.broadcast_to(v, shape).ravel() for k, v in given.items()})
        for p in PROCESS:
            if p not in df:
                df[p] = np.nan
        return self.predict_frame(device, df, state=state, strict=strict)
