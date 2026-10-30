"""Part A dataset generation: reproducible sampling, sharded simulation, resume,
failure logging, assembly and validation.

Reproducibility
---------------
Sample k of a case is drawn from its own RNG block:
    block = k // rng_block,  rng = default_rng([case_seed, block])
and the whole block is always drawn in the same order (VDD first). So sample
k has the same inputs whether you ask for 20, 200 or 10,000 samples, a small
run is an exact prefix of the full run, and topping up after failures just
continues with ids N, N+1, ... Sample ids are therefore stable.

Storage
-------
<out>/_shards/<case>/ids_<start>_<stop>.csv : one file per simulated batch,
    written once, atomically (tmp + rename). Holds sample_id, inputs,
    currents, status and simulator messages (failed rows keep NaN currents).
On restart, every id already present in a valid shard is skipped, so a run
can be interrupted at any time and resumed without duplicating samples.
Final files are assembled once at the end.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .simulator import NgspiceSimulator
from .transistor import (CURRENT_COLS, case_seed, kcl_ok, kcl_residual, nominal_process,
                         simulate_transistors)

SHARD_RE = re.compile(r"ids_(\d+)_(\d+)\.csv$")


def rng_block(cfg: Config) -> int:
    return int(cfg.raw["sampling"].get("rng_block", 1000))


def input_columns(cfg: Config) -> list[str]:
    return [c for c in cfg.columns if c not in CURRENT_COLS]


# ===================================================================== sampling
PROFILES = ("uniform", "targeted")
TARGET_DV_MIN = 1e-9      # V: smallest |Vds| / rail offset drawn by the targeted profile
TARGET_DV_MAX = 0.05      # x VDD: largest


def _near_rail(rng, vdd, size):
    """A terminal voltage near 0 or VDD: offset log-uniform in [1 nV, 0.05 VDD],
    20 % of them exactly on the rail. Returns (values, rail_is_high)."""
    high = rng.uniform(size=size) < 0.5
    off = 10 ** rng.uniform(np.log10(TARGET_DV_MIN), np.log10(TARGET_DV_MAX * vdd), size)
    off = np.where(rng.uniform(size=size) < 0.2, 0.0, off)
    return np.where(high, vdd - off, off), high


def _targeted_vd_vs(rng, vdd, size):
    """Gate-relevant terminal voltages (supplementary data, not the spec distribution):
    40 % near-equal (|Vds| log-uniform 1 nV .. 0.05 VDD, both signs) at a uniform level,
    40 % both terminals near a rail, 20 % one terminal near a rail and the other uniform.
    All random numbers are drawn for every row, so the stream is fixed."""
    mode = rng.choice(3, size=size, p=[0.4, 0.4, 0.2])
    # near-equal
    base = rng.uniform(size=size) * vdd
    dv = 10 ** rng.uniform(np.log10(TARGET_DV_MIN), np.log10(TARGET_DV_MAX * vdd), size)
    dv *= np.where(rng.uniform(size=size) < 0.5, -1.0, 1.0)
    ne_d, ne_s = np.clip(base + dv / 2, 0, vdd), np.clip(base - dv / 2, 0, vdd)
    # rails
    r_d, _ = _near_rail(rng, vdd, size)
    r_s, _ = _near_rail(rng, vdd, size)
    # mixed
    u = rng.uniform(size=size) * vdd
    d_is_rail = rng.uniform(size=size) < 0.5
    m_d, m_s = np.where(d_is_rail, r_d, u), np.where(d_is_rail, u, r_s)
    vd = np.select([mode == 0, mode == 1], [ne_d, r_d], m_d)
    vs = np.select([mode == 0, mode == 1], [ne_s, r_s], m_s)
    return vd, vs


def _draw_block(cfg: Config, case: str, block: int, profile: str = "uniform",
                salt: int = 0) -> pd.DataFrame:
    """salt = 0 reproduces the original spec datasets exactly; any other salt gives an
    independent stream (used for supplementary training data and fresh test sets)."""
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {PROFILES}")
    c = cfg.cases[case]
    nom = nominal_process(cfg, c.device)
    size = rng_block(cfg)
    seed = [case_seed(cfg, case), block] if salt == 0 else [case_seed(cfg, case), block, salt]
    rng = np.random.default_rng(seed)
    vdd = rng.uniform(*cfg.vdd_range, size)                       # VDD first
    temp = rng.uniform(*cfg.temp_range, size)
    w = rng.uniform(*cfg.w_range, size)
    if profile == "uniform":
        vd = rng.uniform(0.0, 1.0, size) * vdd                    # independent U(0, VDD)
        vs = rng.uniform(0.0, 1.0, size) * vdd
    else:
        vd, vs = _targeted_vd_vs(rng, vdd, size)
    df = pd.DataFrame({"vdd": vdd, "temp": temp, "w": w, "vd": vd, "vs": vs,
                       "vg": c.vg_frac * vdd,                     # fixed by the case
                       "vb": c.vb_frac * vdd})
    tol = cfg.process_tolerance
    for p in cfg.process_params:                                  # this device's nominals
        df[p] = nom[p] * rng.uniform(1 - tol, 1 + tol, size)
    df = df.map(lambda x: float(f"{x:.12g}"))                    # = values written to netlist
    if profile == "targeted":                                     # rail values: keep 0 <= V <= VDD
        df["vd"] = np.minimum(np.maximum(df.vd, 0.0), df.vdd)     # after rounding (uniform path is
        df["vs"] = np.minimum(np.maximum(df.vs, 0.0), df.vdd)     # left untouched = original data)
    df.insert(0, "sample_id", np.arange(block * size, (block + 1) * size))
    return df


def generate_samples(cfg: Config, case: str, ids, profile: str = "uniform",
                     salt: int = 0) -> pd.DataFrame:
    """Inputs for the given sample ids (any order/subset), sorted by id."""
    ids = np.unique(np.asarray(ids, dtype=np.int64))
    if len(ids) == 0:
        return pd.DataFrame(columns=["sample_id", *input_columns(cfg)])
    size = rng_block(cfg)
    parts = [_draw_block(cfg, case, b, profile, salt) for b in np.unique(ids // size)]
    df = pd.concat(parts, ignore_index=True)
    return df[df.sample_id.isin(ids)].reset_index(drop=True)


# ===================================================================== shards
def shard_dir(out_dir: Path, case: str) -> Path:
    return Path(out_dir) / "_shards" / case


def _write_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def load_shards(cfg: Config, out_dir: Path, case: str) -> pd.DataFrame:
    """All valid shards of a case. Corrupt/partial shards are ignored (re-simulated)."""
    d = shard_dir(out_dir, case)
    parts = []
    for f in sorted(d.glob("ids_*.csv")) if d.is_dir() else []:
        m = SHARD_RE.search(f.name)
        try:
            s = pd.read_csv(f, dtype={"status": str, "messages": str})
        except Exception:  # noqa: BLE001
            continue
        start, stop = int(m.group(1)), int(m.group(2))
        need = {"sample_id", "status", "messages", *cfg.columns}
        if not need <= set(s.columns) or s.sample_id.min() < start or s.sample_id.max() >= stop:
            continue
        parts.append(s)
    if not parts:
        return pd.DataFrame(columns=["sample_id", *cfg.columns, "status", "messages"])
    df = pd.concat(parts, ignore_index=True)
    df["messages"] = df["messages"].fillna("")
    return df.drop_duplicates("sample_id", keep="first").sort_values("sample_id",
                                                                     ignore_index=True)


@dataclass
class CaseRun:
    case: str
    simulated_now: int = 0
    seconds: float = 0.0


def simulate_ids(cfg: Config, case: str, ids, out_dir: Path, sim: NgspiceSimulator,
                 batch_size: int, bar=None, profile: str = "uniform", salt: int = 0) -> CaseRun:
    """Simulate the given ids in batches; each batch is saved as one shard."""
    import time
    run = CaseRun(case)
    ids = np.sort(np.asarray(ids, dtype=np.int64))
    t0 = time.perf_counter()
    for i in range(0, len(ids), batch_size):
        chunk_ids = ids[i:i + batch_size]
        inputs = generate_samples(cfg, case, chunk_ids, profile, salt)
        state = {"left": len(inputs)}

        def prog(done, total, elapsed, _s=state):
            if bar is not None:
                step = min(sim.chunk, _s["left"])
                _s["left"] -= step
                bar.update(step)

        res = simulate_transistors(cfg, case, inputs, sim, progress=prog)
        if bar is not None and state["left"] > 0:
            bar.update(state["left"])
        warnings = res.attrs.get("batch_warnings", {})
        res = res[["sample_id", *cfg.columns, "status", "messages"]]
        name = f"ids_{int(chunk_ids[0]):07d}_{int(chunk_ids[-1]) + 1:07d}"
        # warnings first: a shard CSV only exists once its sidecar is written
        sidecar = shard_dir(out_dir, case) / f"{name}.warnings.json"
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        tmp = sidecar.with_suffix(".tmp")
        tmp.write_text(json.dumps(warnings, indent=1), encoding="utf-8")
        os.replace(tmp, sidecar)
        _write_atomic(res, shard_dir(out_dir, case) / f"{name}.csv")
        run.simulated_now += len(res)
        if bar is not None:
            bar.set_postfix(failed=int((res.status != "ok").sum()), refresh=False)
    run.seconds = time.perf_counter() - t0
    return run


def shard_warnings(out_dir: Path, case: str) -> dict:
    """Sum of simulator warning counts over all shards of a case (from the
    .warnings.json sidecars). 'shards_without_record' counts older shards."""
    from collections import Counter
    total, missing = Counter(), 0
    d = shard_dir(out_dir, case)
    for f in sorted(d.glob("ids_*.csv")) if d.is_dir() else []:
        side = f.with_name(f.stem + ".warnings.json")
        if side.is_file():
            total.update(json.loads(side.read_text(encoding="utf-8")))
        else:
            missing += 1
    return {"counts": dict(total.most_common()), "shards_without_record": missing}


class TooManyFailures(RuntimeError):
    pass


def ids_needed(shards: pd.DataFrame, n_valid: int, max_extra: int) -> np.ndarray:
    """Ids still to simulate so that at least n_valid samples succeed.

    1. The base range 0..n_valid-1 is completed first.
    2. If F samples failed, the next F unused ids are simulated (top-up); repeat.
    Raises TooManyFailures if more than max_extra top-up ids would be needed."""
    done = set(shards.sample_id.astype(int)) if len(shards) else set()
    base = [i for i in range(n_valid) if i not in done]
    if base:
        return np.array(base, dtype=np.int64)
    deficit = n_valid - int((shards.status == "ok").sum())
    if deficit <= 0:
        return np.array([], dtype=np.int64)
    start = max(done) + 1
    if start + deficit - n_valid > max_extra:
        raise TooManyFailures(f"more than {max_extra} extra samples needed; "
                              "inspect the failure log before continuing")
    return np.arange(start, start + deficit, dtype=np.int64)


# ===================================================================== assembly
def assemble(cfg: Config, shards: pd.DataFrame, n_valid: int):
    """(final, index, failures): first n_valid successful samples by id."""
    ok = shards[shards.status == "ok"].sort_values("sample_id")
    final = ok.head(n_valid).reset_index(drop=True)
    last_id = int(final.sample_id.max()) if len(final) else -1
    failures = shards[(shards.status != "ok") & (shards.sample_id <= max(last_id, n_valid - 1))]
    index = pd.DataFrame({"row": np.arange(len(final)), "sample_id": final.sample_id.astype(int),
                          "kcl_sum": kcl_residual(final)})
    return final[list(cfg.columns)], index, failures.reset_index(drop=True)


# ===================================================================== validation
def validate(cfg: Config, case: str, df: pd.DataFrame, n_expected: int,
             sample_ids: pd.Series | None = None,
             allow_exact_zero: bool = False) -> list[tuple[bool, str]]:
    """Checks on a final dataset (columns exactly cfg.columns).

    allow_exact_zero: the targeted profile places terminals exactly on the rails, where a
    current can be exactly 0 in SPICE (e.g. Ig when Vgd = 0 and Vgs ~ 0). Then exact zeros
    are reported, not failed. (Failed simulations never appear as zeros either way: they
    are excluded from the final data and logged in failures.csv.)"""
    c = cfg.cases[case]
    nom = nominal_process(cfg, c.device)
    eps = 1e-9
    num = df[list(cfg.columns)].to_numpy(float) if len(df) else np.zeros((0, len(cfg.columns)))
    lo, hi = cfg.vdd_range
    tol = cfg.process_tolerance
    checks = [
        (len(df) == n_expected, f"row count {len(df)} == {n_expected}"),
        (list(df.columns) == list(cfg.columns), "column order " + ", ".join(cfg.columns)),
        (bool(np.isfinite(num).all()), "all values finite (no NaN/inf)"),
        (bool(df.vdd.between(lo - eps, hi + eps).all()), f"vdd in [{lo:g}, {hi:g}] V"),
        (bool(df.temp.between(cfg.temp_range[0] - eps, cfg.temp_range[1] + eps).all()),
         f"temp in [{cfg.temp_range[0]:g}, {cfg.temp_range[1]:g}] C"),
        (bool(df.w.between(cfg.w_range[0] * (1 - eps), cfg.w_range[1] * (1 + eps)).all()),
         f"w in [{cfg.w_range[0]*1e9:g}, {cfg.w_range[1]*1e9:g}] nm"),
        (bool(((df.vd >= -eps) & (df.vd <= df.vdd + eps)).all()), "0 <= vd <= vdd"),
        (bool(((df.vs >= -eps) & (df.vs <= df.vdd + eps)).all()), "0 <= vs <= vdd"),
        (bool(np.allclose(df.vg, c.vg_frac * df.vdd, rtol=1e-9, atol=1e-12)),
         f"vg == {c.vg_frac:g}*vdd"),
        (bool(np.allclose(df.vb, c.vb_frac * df.vdd, rtol=1e-9, atol=1e-12)),
         f"vb == {c.vb_frac:g}*vdd"),
    ]
    for p in cfg.process_params:
        r = df[p] / nom[p]
        checks.append((bool(r.between(1 - tol - eps, 1 + tol + eps).all()),
                       f"{p} within +/-{tol:.0%} of {c.device} nominal {nom[p]:g}"))
    inp = [x for x in cfg.columns if x not in CURRENT_COLS]
    checks.append((not df.duplicated(subset=inp).any(), "input rows unique"))
    if sample_ids is not None:
        checks.append((sample_ids.is_unique, "sample ids unique"))
    n_zero = int((df[CURRENT_COLS] == 0).any(axis=1).sum())
    checks.append((n_zero == 0 or allow_exact_zero,
                   "no exactly-zero currents" if n_zero == 0 else
                   f"{n_zero} rows with an exactly-zero SPICE current"
                   + (" (allowed: targeted rail biases)" if allow_exact_zero else "")))
    checks.append((bool(kcl_ok(df).all()),
                   f"Kirchhoff |id+ig+is+ib| within tolerance for all rows "
                   f"(max {kcl_residual(df).abs().max():.2e} A)" if len(df) else "Kirchhoff"))
    return checks


def current_stats(df: pd.DataFrame) -> dict:
    out = {}
    for col in CURRENT_COLS:
        a = df[col].abs()
        out[col] = {"min_abs": float(a.min()), "median_abs": float(a.median()),
                    "max_abs": float(a.max()), "frac_negative": float((df[col] < 0).mean())}
    return out


def fmt_seconds(s: float) -> str:
    return f"{s:.1f} s" if s < 120 else f"{s/60:.1f} min"

