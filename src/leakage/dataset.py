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
def _draw_block(cfg: Config, case: str, block: int) -> pd.DataFrame:
    c = cfg.cases[case]
    nom = nominal_process(cfg, c.device)
    size = rng_block(cfg)
    rng = np.random.default_rng([case_seed(cfg, case), block])
    vdd = rng.uniform(*cfg.vdd_range, size)                       # VDD first
    df = pd.DataFrame({
        "vdd": vdd,
        "temp": rng.uniform(*cfg.temp_range, size),
        "w": rng.uniform(*cfg.w_range, size),
        "vd": rng.uniform(0.0, 1.0, size) * vdd,                  # independent U(0, VDD)
        "vs": rng.uniform(0.0, 1.0, size) * vdd,
        "vg": c.vg_frac * vdd,                                    # fixed by the case
        "vb": c.vb_frac * vdd,
    })
    tol = cfg.process_tolerance
    for p in cfg.process_params:                                  # this device's nominals
        df[p] = nom[p] * rng.uniform(1 - tol, 1 + tol, size)
    df = df.map(lambda x: float(f"{x:.12g}"))                    # = values written to netlist
    df.insert(0, "sample_id", np.arange(block * size, (block + 1) * size))
    return df


def generate_samples(cfg: Config, case: str, ids) -> pd.DataFrame:
    """Inputs for the given sample ids (any order/subset), sorted by id."""
    ids = np.unique(np.asarray(ids, dtype=np.int64))
    if len(ids) == 0:
        return pd.DataFrame(columns=["sample_id", *input_columns(cfg)])
    size = rng_block(cfg)
    parts = [_draw_block(cfg, case, b) for b in np.unique(ids // size)]
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
                 batch_size: int, bar=None) -> CaseRun:
    """Simulate the given ids in batches; each batch is saved as one shard."""
    import time
    run = CaseRun(case)
    ids = np.sort(np.asarray(ids, dtype=np.int64))
    t0 = time.perf_counter()
    for i in range(0, len(ids), batch_size):
        chunk_ids = ids[i:i + batch_size]
        inputs = generate_samples(cfg, case, chunk_ids)
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
             sample_ids: pd.Series | None = None) -> list[tuple[bool, str]]:
    """Checks on a final dataset (columns exactly cfg.columns)."""
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
    checks.append((bool((df[CURRENT_COLS] != 0).all().all()), "no exactly-zero currents"))
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

