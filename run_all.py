"""Master runner for the whole project (run from the project root with the venv's Python).

    python run_all.py --list                 # show all stages
    python run_all.py --mode verify          # tests + deliverable audit (~1 min)
    python run_all.py --mode smoke           # small end-to-end check of every stage (~10 min)
    python run_all.py --mode full            # regenerate everything in order (many hours; resumable)
    python run_all.py --mode full --from b3  # continue from a stage
    python run_all.py --mode full --only a1_data_check d4_report
    python run_all.py --mode full --dry-run  # print the commands only

* Every command uses paths relative to the project root and the current Python interpreter.
* Logs: results/run_logs/<mode>_<stage>.log; summary: results/run_logs/<mode>_summary.md
* One-time evaluations (A2 test, A2b fresh test, final improvement evaluation) refuse to run a
  second time (exit code 2); the runner reports those as "skipped (already evaluated)".
* Long stages are marked; generation stages are resumable (shards), training stages cache results.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
LOGS = ROOT / "results" / "run_logs"
CASES = ["nmos_on", "nmos_off", "pmos_on", "pmos_off"]


def gen(*args):
    return [PY, "scripts/a1_generate_datasets.py", "--skip-smoke", "--no-progress", *args]


def copy_models(src: str, cases: list[str]):
    def run():
        for c in cases:
            shutil.copy2(ROOT / "trained_models" / src / f"{c}.joblib", ROOT / "trained_models" / f"{c}.joblib")
        return 0
    return run


# (name, description, commands or callable, est. time, one_time)
FULL = [
    ("env", "environment check (Python, packages, ngspice, model file)", [[PY, "scripts/check_environment.py"]], "10 s", False),
    ("inventory", "extract + inventory supplied files", [[PY, "scripts/inventory_inputs.py"]], "5 s", False),
    ("a1_engine", "transistor simulation engine smoke test + example netlists", [[PY, "scripts/a1_smoke_test.py"]], "10 s", False),
    ("a1_datasets", "4 x 10,000 transistor datasets (resumable)", [[PY, "scripts/a1_generate_datasets.py", "--no-progress"]], "5 min", False),
    ("a1_data_check", "Kirchhoff check, trend plots, data-quality report", [[PY, "scripts/a1_data_check.py"]], "2 min", False),
    ("a2_train", "4 algorithms x 4 cases, tuning + selection (cached)", [[PY, "scripts/a2_train_models.py"]], "1.5 h", False),
    ("a2_test", "ONE-TIME A2 test evaluation", [[PY, "scripts/a2_evaluate_test.py"]], "1 min", True),
    ("a2b_data", "targeted training data + fresh test sets",
     [gen("--profile", "targeted", "--seed-salt", "1", "--n", "4000", "--cases", "nmos_on", "pmos_on", "--out", "data/transistor_aug_on"),
      gen("--profile", "targeted", "--seed-salt", "1", "--n", "2000", "--cases", "nmos_off", "pmos_off", "--out", "data/transistor_aug_off"),
      gen("--profile", "uniform", "--seed-salt", "2", "--n", "1500", "--out", "data/transistor_test2_uniform"),
      gen("--profile", "targeted", "--seed-salt", "3", "--n", "1000", "--out", "data/transistor_test2_targeted")], "3 min", False),
    ("a2b_train", "v2 models (signed-log Vds feature + targeted data)", [[PY, "scripts/a2b_train_v2.py"]], "3 h", False),
    ("a2b_test", "ONE-TIME fresh-test evaluation of v1/v2", [[PY, "scripts/a2b_evaluate_test2.py"]], "1 min", True),
    ("a2b_amend", "documented amendment: OFF models -> v2 (results/a2b/selection_amendment.json)",
     copy_models("v2", ["nmos_off", "pmos_off"]), "1 s", False),
    ("b3", "gate SPICE references, 18 cases x 200 corners", [[PY, "scripts/b3_gate_spice.py"]], "12 min", False),
    ("b4", "hierarchical ML gate leakage (Part B steps 4-5)", [[PY, "scripts/b4_gate_ml.py"]], "10 s", False),
    ("c1", "CLA3 full-circuit SPICE references, Cin = 0 half: 64 x 200", [[PY, "scripts/c1_cla_spice.py"]], "16 min", False),
    ("c2", "CLA3 hierarchical ML + evaluation", [[PY, "scripts/c2_cla_ml.py"]], "1 min", False),
    ("d1_diagnostics", "improvement phase: error diagnostics (train/val)", [[PY, "scripts/d1_diagnostics.py"]], "2 min", False),
    ("d_final_eval_data", "independent final-evaluation data (new corners + fresh transistor tests)",
     [gen("--profile", "uniform", "--seed-salt", "4", "--n", "1500", "--out", "data/final_eval/transistor_uniform"),
      gen("--profile", "targeted", "--seed-salt", "5", "--n", "1000", "--out", "data/final_eval/transistor_targeted"),
      [PY, "scripts/b3_gate_spice.py", "--corner-seed", "5005", "--out-data", "data/final_eval/gates",
       "--out-results", "results/improve/final_eval_b3", "--no-netlists"],
      [PY, "scripts/c1_cla_spice.py", "--corners-file", "data/final_eval/gates/corners.csv",
       "--out-data", "data/final_eval/circuit", "--out-results", "results/improve/final_eval_c", "--no-netlists"]],
     "35 min", False),
    ("d_aug2", "extra targeted training data (salt 6)",
     [gen("--profile", "targeted", "--seed-salt", "6", "--n", "4000", "--cases", "nmos_on", "pmos_on", "--out", "data/transistor_aug2_on"),
      gen("--profile", "targeted", "--seed-salt", "6", "--n", "2000", "--cases", "nmos_off", "pmos_off", "--out", "data/transistor_aug2_off")],
     "3 min", False),
    ("d2_experiments", "improvement experiments E0..E6 (PyTorch L-BFGS; cached per case)",
     [[PY, "scripts/d2_experiments.py", "--exp", "E0_baseline", "E1_ens_v2cache", "E2_torch_same", "--max-iter", "15000"],
      [PY, "scripts/d2_experiments.py", "--exp", "E5_torch_moredata", "--max-iter", "20000"],
      [PY, "scripts/d2_experiments.py", "--exp", "E6_ens_E2_E5", "--select", "--summary"]], "8-10 h", False),
    ("d5_op_check", "SPICE operating-point uniqueness check of all references",
     [[PY, "scripts/d5_op_uniqueness.py", "--gates", "data/final_eval/gates", "data/gates"],
      [PY, "scripts/d5_op_uniqueness.py", "--cla", "data/final_eval/circuit:data/final_eval/gates", "data/circuit:data/gates"]],
     "35 min", False),
    ("d3_final_eval", "ONE-TIME before/after evaluation on independent data",
     [[PY, "scripts/d3_final_eval.py"], [PY, "scripts/d3_final_eval.py", "--refs", "corrected"]], "3 min", True),
    ("d4_report", "improvement report", [[PY, "scripts/d4_report.py"]], "10 s", False),
    ("promote_v3", "final models = trained_models/v3", copy_models("v3", CASES), "1 s", False),
    ("c1_cin1", "CLA3 Cin = 1 half (vectors 64..127) at both corner sets + operating-point check",
     [[PY, "scripts/c1_cla_spice.py", "--cin", "1"],
      [PY, "scripts/c1_cla_spice.py", "--cin", "1", "--corners-file", "data/final_eval/gates/corners.csv",
       "--out-data", "data/final_eval/circuit", "--out-results", "results/improve/final_eval_c", "--no-netlists"],
      [PY, "scripts/d5_op_uniqueness.py", "--cla", "data/circuit:data/gates:_cin1",
       "data/final_eval/circuit:data/final_eval/gates:_cin1"]], "50 min", False),
    ("z1_deliverables", "final-model deliverable tables and figures", [[PY, "scripts/z1_final_deliverables.py"]], "1 min", False),
    ("z3_report", "final technical report (Markdown + HTML)", [[PY, "scripts/z3_build_report.py"]], "10 s", False),
    ("tests", "unit / integration tests", [[PY, "-m", "pytest", "-q", "tests"]], "1 min", False),
    ("audit", "deliverable audit + checklist", [[PY, "scripts/z2_audit.py"]], "1 min", False),
]

SMOKE = [
    ("env", "environment check", [[PY, "scripts/check_environment.py"]], "10 s", False),
    ("a1_engine", "transistor engine smoke test", [[PY, "scripts/a1_smoke_test.py"]], "10 s", False),
    ("a1_dataset_smoke", "20-sample dataset smoke test per case", [[PY, "scripts/a1_generate_datasets.py", "--smoke"]], "10 s", False),
    ("a1_dataset_small", "200-sample dataset per case (data/transistor_small)",
     [[PY, "scripts/a1_generate_datasets.py", "--small", "--skip-smoke", "--no-progress"]], "15 s", False),
    ("a2_quick", "ML pipeline with a tiny grid (results/a2_quick; saves no models)", [[PY, "scripts/a2_train_models.py", "--quick"]], "2 min", False),
    ("predict", "reload final models and predict one transistor vs SPICE",
     [[PY, "scripts/a2_predict.py", "--device", "nmos", "--vd", "1.0", "--vg", "0", "--vs", "0", "--vb", "0",
       "--vdd", "1.0", "--temp", "27", "--w", "90e-9", "--spice"]], "5 s", False),
    ("b3_smoke", "5 gates, 18 cases, nominal corner + corner 0", [[PY, "scripts/b3_gate_spice.py", "--smoke"]], "15 s", False),
    ("c1_smoke", "CLA3, all 128 vectors (Cin = 0 and 1) at nominal corner + corner 0",
     [[PY, "scripts/c1_cla_spice.py", "--smoke"], [PY, "scripts/c1_cla_spice.py", "--smoke", "--cin", "1"]], "40 s", False),
    ("tests", "unit / integration tests", [[PY, "-m", "pytest", "-q", "tests"]], "1 min", False),
    ("audit", "deliverable audit + checklist", [[PY, "scripts/z2_audit.py"]], "1 min", False),
]

VERIFY = [s for s in SMOKE if s[0] in ("env", "tests", "audit")]


def run_stage(mode, name, desc, cmds, one_time, dry):
    log = LOGS / f"{mode}_{name}.log"
    if callable(cmds):
        if dry:
            print(f"   (python step: {desc})")
            return "dry-run", 0.0
        t0 = time.perf_counter()
        rc = cmds()
        return ("ok" if rc == 0 else f"failed ({rc})"), time.perf_counter() - t0
    t0 = time.perf_counter()
    with log.open("w", encoding="utf-8") as f:
        for cmd in cmds:
            shown = " ".join(c if c != PY else "python" for c in cmd)
            print(f"   $ {shown}")
            if dry:
                continue
            f.write(f"$ {shown}\n")
            f.flush()
            r = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                               env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
            if r.returncode == 2 and one_time:
                return "skipped (already evaluated)", time.perf_counter() - t0
            if r.returncode != 0:
                return f"FAILED (exit {r.returncode}, see {log.relative_to(ROOT)})", time.perf_counter() - t0
    return ("dry-run" if dry else "ok"), time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", choices=["verify", "smoke", "full"], default="smoke")
    ap.add_argument("--from", dest="start", default=None)
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep-going", action="store_true", help="continue after a failed stage")
    a = ap.parse_args()
    stages = {"verify": VERIFY, "smoke": SMOKE, "full": FULL}[a.mode]
    if a.list:
        for m, st in (("verify", VERIFY), ("smoke", SMOKE), ("full", FULL)):
            print(f"\n[{m}]")
            for n, d, _, t, ot in st:
                print(f"  {n:18s} {t:>8s}  {d}{'  (one-time)' if ot else ''}")
        return 0
    names = [s[0] for s in stages]
    if a.start:
        stages = stages[names.index(a.start):]
    if a.only:
        stages = [s for s in stages if s[0] in a.only]
    LOGS.mkdir(parents=True, exist_ok=True)
    results = []
    for name, desc, cmds, est, one_time in stages:
        print(f"\n== [{a.mode}] {name}: {desc}  (est. {est})")
        status, secs = run_stage(a.mode, name, desc, cmds, one_time, a.dry_run)
        print(f"   -> {status}  ({secs:.0f} s)")
        results.append((name, status, secs))
        if status.startswith("FAILED") and not a.keep_going:
            break
    lines = [f"# run_all.py --mode {a.mode} ({time.strftime('%Y-%m-%d %H:%M:%S')})", "",
             "| stage | status | seconds |", "|---|---|---|"] + [f"| {n} | {s} | {t:.0f} |" for n, s, t in results]
    if not a.dry_run:
        (LOGS / f"{a.mode}_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    return 0 if all(not s.startswith("FAILED") for _, s, _ in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
