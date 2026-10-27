"""Part A, step 1: generate the four single-transistor PVT datasets with ngspice.

Modes
-----
  --smoke   20 samples per case, simulated in memory and validated (no files
            except results/a1_dataset_smoke.csv). Run this after code changes.
  --small   200 samples per case into data/transistor_small/ (quick end-to-end test).
  (default) full run: N = config [sampling].n_samples (10,000) valid samples per
            case into data/transistor/. A 20-sample smoke test runs first.

Resuming: just rerun the same command. Ids already saved in shards are skipped.

Outputs (in the output folder)
  <case>.csv          final dataset, columns exactly as the spec, N rows, sorted by sample_id
  <case>_index.csv    row -> stable sample_id, plus the Kirchhoff sum of that row
  failures.csv        every failed simulation (case, sample_id, inputs, simulator messages)
  metadata.json       reproducibility info: seeds, ranges, versions, hashes, timings, checks
  summary.md          human-readable summary (also copied to reports/ for the full run)
  _shards/<case>/     intermediate per-batch results (used for resume)

Examples (PowerShell, from the project root)
  .\\.venv\\Scripts\\python.exe scripts\\a1_generate_datasets.py --smoke
  .\\.venv\\Scripts\\python.exe scripts\\a1_generate_datasets.py --small
  .\\.venv\\Scripts\\python.exe scripts\\a1_generate_datasets.py
  .\\.venv\\Scripts\\python.exe scripts\\a1_generate_datasets.py --cases nmos_off --n 2000 --out data\\tmp
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from leakage.config import load_config  # noqa: E402
from leakage.dataset import (TooManyFailures, assemble, current_stats, fmt_seconds,  # noqa: E402
                             generate_samples, ids_needed, load_shards, rng_block,
                             shard_warnings, simulate_ids, validate)
from leakage.simulator import NgspiceSimulator  # noqa: E402
from leakage.spice import ngspice_version  # noqa: E402
from leakage.transistor import (CURRENT_COLS, case_seed, kcl_residual,  # noqa: E402
                                simulate_transistors)

SMOKE_N = 20
SMALL_N = 200


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_info() -> dict:
    def run(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True,
                                  timeout=20).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def pkg_versions() -> dict:
    import importlib
    out = {"python": platform.python_version()}
    for m in ("numpy", "pandas", "tqdm"):
        out[m] = getattr(importlib.import_module(m), "__version__", "?")
    return out


def print_checks(checks, indent="    ") -> bool:
    for ok, text in checks:
        print(f"{indent}[{'PASS' if ok else 'FAIL'}] {text}")
    return all(ok for ok, _ in checks)


# ===================================================================== smoke
def run_smoke(cfg, cases, sim) -> bool:
    print(f"== Smoke test: {SMOKE_N} samples per case (sample ids 0..{SMOKE_N - 1}) ==")
    all_ok, rows = True, []
    for case in cases:
        inp = generate_samples(cfg, case, range(SMOKE_N))
        t0 = time.perf_counter()
        res = simulate_transistors(cfg, case, inp, sim)
        dt_s = time.perf_counter() - t0
        n_fail = int((res.status != "ok").sum())
        ok_rows = res[res.status == "ok"]
        checks = [(n_fail == 0, f"{SMOKE_N - n_fail}/{SMOKE_N} simulations succeeded")]
        checks += validate(cfg, case, ok_rows[list(cfg.columns)].reset_index(drop=True),
                           SMOKE_N, ok_rows.sample_id)
        print(f"  {case}: {dt_s:.2f} s, |Id| {ok_rows['id'].abs().min():.2e} .. "
              f"{ok_rows['id'].abs().max():.2e} A")
        all_ok &= print_checks(checks)
        if n_fail:
            for _, r in res[res.status != "ok"].iterrows():
                print(f"    failed sample {r.sample_id}: {r.messages}")
        res.insert(0, "case", case)
        rows.append(res)
    cfg.results.mkdir(parents=True, exist_ok=True)
    pd.concat(rows).to_csv(cfg.results / "a1_dataset_smoke.csv", index=False)
    print(f"  -> results/a1_dataset_smoke.csv\nSMOKE TEST {'PASSED' if all_ok else 'FAILED'}\n")
    return all_ok


# ===================================================================== generation
def generate_case(cfg, case, n, out, sim, batch, max_extra, progress) -> dict:
    t0 = time.perf_counter()
    simulated_now = 0
    while True:
        shards = load_shards(cfg, out, case)
        todo = ids_needed(shards, n, max_extra)
        if len(todo) == 0:
            break
        label = f"{case:8s}" + (" (top-up)" if todo.min() >= n else "")
        with tqdm(total=len(todo), desc=label, unit="sim", disable=not progress,
                  dynamic_ncols=True, initial=0) as bar:
            if len(shards):
                bar.write(f"  {case}: resuming, {len(shards)} samples already saved, "
                          f"{len(todo)} to simulate")
            run = simulate_ids(cfg, case, todo, out, sim, batch, bar)
        simulated_now += run.simulated_now
    shards = load_shards(cfg, out, case)
    return {"shards": shards, "simulated_now": simulated_now,
            "seconds_now": time.perf_counter() - t0}


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--smoke", action="store_true", help="only the 20-sample smoke test")
    ap.add_argument("--small", action="store_true", help=f"{SMALL_N} samples/case into data/transistor_small")
    ap.add_argument("--n", type=int, default=None, help="valid samples per case")
    ap.add_argument("--out", type=Path, default=None, help="output folder")
    ap.add_argument("--cases", nargs="+", default=list(cfg.cases), choices=list(cfg.cases))
    ap.add_argument("--batch", type=int, default=int(cfg.raw["sampling"].get("batch_size", 1000)),
                    help="samples per saved shard")
    ap.add_argument("--workers", type=int, default=cfg.ngspice_workers,
                    help="parallel ngspice processes")
    ap.add_argument("--max-extra", type=int, default=None,
                    help="max extra samples drawn to replace failures (default 5%% of n, >= 50)")
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--no-progress", action="store_true")
    a = ap.parse_args()

    n = a.n or (SMALL_N if a.small else cfg.n_samples)
    out = (a.out or (cfg.data / ("transistor_small" if a.small else "transistor"))).resolve()
    workers = max(1, min(a.workers, 64))
    batch = max(1, a.batch)
    max_extra = a.max_extra if a.max_extra is not None else max(50, math.ceil(0.05 * n))
    sim = NgspiceSimulator.from_config(cfg, workers=workers,
                                       chunk=max(1, math.ceil(batch / workers)))
    ver = ngspice_version(sim.exe)
    model_hash = sha256(cfg.spice_model_file)
    if model_hash != cfg.spice_model_sha256:
        print(f"ERROR: {cfg.spice_model_file} hash {model_hash} != config; aborting")
        return 2

    print(f"Model {cfg.model_name} ({cfg.spice_model_file.name}), VDD {cfg.vdd_nominal} V, "
          f"L {cfg.lmin*1e9:.0f} nm | ngspice {ver} | {workers} workers x chunk {sim.chunk}")

    if a.smoke:
        return 0 if run_smoke(cfg, a.cases, sim) else 1
    if not a.skip_smoke and not run_smoke(cfg, a.cases, sim):
        print("Smoke test failed: fix it before generating data.")
        return 1

    print(f"== Generating {n} valid samples per case -> {out} (batch {batch}) ==")
    t_all = time.perf_counter()
    results = {}
    try:
        for case in a.cases:
            results[case] = generate_case(cfg, case, n, out, sim, batch, max_extra,
                                          progress=not a.no_progress)
    except KeyboardInterrupt:
        print("\nInterrupted. Everything simulated so far is saved; rerun the same command to resume.")
        return 130
    except TooManyFailures as e:
        print(f"\nSTOPPED: {e}")
        return 3
    wall = time.perf_counter() - t_all

    # ---------------------------------------------------------------- assemble + validate
    print("\n== Assembling and validating ==")
    case_meta, all_fail, all_ok = {}, [], True
    for case in a.cases:
        shards = results[case]["shards"]
        final, index, failures = assemble(cfg, shards, n)
        csv_path, idx_path = out / f"{case}.csv", out / f"{case}_index.csv"
        final.to_csv(csv_path, index=False)
        index.to_csv(idx_path, index=False)
        checks = validate(cfg, case, final, n, index.sample_id)
        print(f"  {case}:")
        ok = print_checks(checks)
        all_ok &= ok
        if len(failures):
            f = failures.copy()
            f.insert(0, "case", case)
            all_fail.append(f)
        k = kcl_residual(final).abs()
        case_meta[case] = {
            "device": cfg.cases[case].device,
            "vg": f"{cfg.cases[case].vg_frac:g}*vdd", "vb": f"{cfg.cases[case].vb_frac:g}*vdd",
            "seed": case_seed(cfg, case),
            "rows": len(final),
            "sample_id_range": [int(index.sample_id.min()), int(index.sample_id.max())] if len(index) else None,
            "simulated_total": int(len(shards)),
            "succeeded_total": int((shards.status == "ok").sum()),
            "failed_total": int((shards.status != "ok").sum()),
            "simulated_this_run": results[case]["simulated_now"],
            "seconds_this_run": round(results[case]["seconds_now"], 2),
            "simulator_warnings": shard_warnings(out, case),
            "kcl_abs_max": float(k.max()), "kcl_abs_median": float(k.median()),
            "currents": current_stats(final),
            "checks": [{"pass": bool(p), "check": t} for p, t in checks],
            "csv": csv_path.name, "csv_sha256": sha256(csv_path),
        }

    fail_df = (pd.concat(all_fail, ignore_index=True) if all_fail else
               pd.DataFrame(columns=["case", "sample_id", *cfg.columns, "status", "messages"]))
    fail_df.to_csv(out / "failures.csv", index=False)

    total_sim = sum(m["simulated_total"] for m in case_meta.values())
    total_ok = sum(m["succeeded_total"] for m in case_meta.values())
    total_now = sum(m["simulated_this_run"] for m in case_meta.values())
    meta = {
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
        "git": git_info(), "versions": pkg_versions(),
        "platform": platform.platform(),
        "ngspice": {"version": ver, "path": str(sim.exe), "options": cfg.spice_options},
        "model": {"name": cfg.model_name, "file": str(cfg.spice_model_file.relative_to(cfg.root)),
                  "sha256": model_hash},
        "assignment": {"vdd_nominal": cfg.vdd_nominal, "lmin": cfg.lmin,
                       "test_circuit": cfg.test_circuit},
        "sampling": {"n_valid_per_case": n, "vdd_range": cfg.vdd_range,
                     "temp_range_c": cfg.temp_range, "w_range_m": cfg.w_range,
                     "vd_vs": "independent U(0, vdd)", "process_params": cfg.process_params,
                     "process_tolerance": cfg.process_tolerance, "rng_block": rng_block(cfg),
                     "rng": "numpy default_rng([case_seed, sample_id // rng_block]); VDD drawn first",
                     "rounding": "inputs rounded to 12 significant digits (= netlist values)",
                     "top_up": "failed samples are not replaced by zeros; extra ids are simulated "
                               "until n valid samples exist; final = first n valid by sample_id"},
        "conventions": {"current_sign": "positive = flowing INTO the device terminal (I = -i(Vsource))",
                        "units": "V, degC, m (w), vth0 V, toxe m, ndep cm^-3, u0 m^2/Vs, xj m, A"},
        "run": {"workers": workers, "chunk": sim.chunk, "batch": batch,
                "wall_seconds_this_run": round(wall, 2), "simulated_this_run": total_now},
        "totals": {"simulated": total_sim, "succeeded": total_ok, "failed": total_sim - total_ok},
        "cases": case_meta,
        "all_checks_passed": bool(all_ok),
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # ---------------------------------------------------------------- summary
    lines = ["# Part A1 dataset generation summary", "",
             f"- Created: {meta['created']}  |  command: `{meta['command']}`",
             f"- Model `{meta['model']['file']}` (sha256 {model_hash[:12]}...), ngspice {ver}",
             f"- Git commit {meta['git']['commit'][:10]}{' (dirty)' if meta['git']['dirty'] else ''}",
             f"- Output folder: `{out.relative_to(cfg.root) if out.is_relative_to(cfg.root) else out}`",
             f"- This run: {total_now} simulations in {fmt_seconds(wall)}"
             + (f" ({total_now / wall:.0f} sim/s)" if total_now and wall > 0 else ""), "",
             "| Case | Rows | Simulated | Succeeded | Failed | Sample ids | max abs KCL sum (A) | median abs Id (A) | Checks |",
             "|---|---:|---:|---:|---:|---|---:|---:|---|"]
    for case, m in case_meta.items():
        rng = m["sample_id_range"]
        lines.append(f"| {case} | {m['rows']} | {m['simulated_total']} | {m['succeeded_total']} | "
                     f"{m['failed_total']} | {rng[0]}..{rng[1]} | {m['kcl_abs_max']:.2e} | "
                     f"{m['currents']['id']['median_abs']:.2e} | "
                     f"{'all pass' if all(c['pass'] for c in m['checks']) else 'FAIL'} |")
    lines += ["", "Simulator warnings (from successful batches; counts per message):", ""]
    for case, m in case_meta.items():
        w = m["simulator_warnings"]
        txt = "; ".join(f"{msg} ×{n}" for msg, n in list(w["counts"].items())[:3]) or "none"
        if w["shards_without_record"]:
            txt += f" (+{w['shards_without_record']} older shards without a warning record)"
        lines.append(f"- {case}: {txt}")
    lines += ["", f"**Totals:** {total_sim} simulated, {total_ok} succeeded, "
              f"{total_sim - total_ok} failed (see `failures.csv`).",
              f"**Validation:** {'all checks passed' if all_ok else 'SOME CHECKS FAILED'}.", ""]
    summary = "\n".join(lines)
    (out / "summary.md").write_text(summary, encoding="utf-8")
    if out == (cfg.data / "transistor").resolve():
        (cfg.reports / "a1_dataset_summary.md").write_text(summary, encoding="utf-8")

    print("\n" + summary)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
