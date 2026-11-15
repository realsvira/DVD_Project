"""Deliverable audit: verifies every item of the project PDF's deliverables list (Section 6)
by CONTENT, not just by file existence, and writes reports/deliverables_checklist.md.

Checks include: dataset columns/row counts/finiteness; every example netlist actually runs in
ngspice and contains the mandatory .options line; Kirchhoff summary; trend plots; the 4x4 model
comparison; final models load and predict; transistor test tables/plots; the 18-case gate table;
gate and circuit parity plots; SPICE reference validity; code, seeds, runner, README, install
guide, requirements and the final report.

Run:  .venv/bin/python scripts/z2_audit.py      (exit code 0 = everything verified)
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from leakage.config import load_config  # noqa: E402
from leakage.spice import find_ngspice  # noqa: E402

CFG = load_config()
CASES = list(CFG.cases)
COLS = ["vdd", "temp", "w", "vd", "vs", "vg", "vb", "vth0", "toxe", "ndep", "u0", "xj", "id", "ig", "is", "ib"]
ITEMS: list[dict] = []


def item(req: str, files: list[str], checks: list[tuple[bool, str]]):
    ok = all(c for c, _ in checks)
    ITEMS.append({"req": req, "files": files, "ok": ok, "checks": checks})
    print(f"[{'PASS' if ok else 'FAIL'}] {req}")
    for c, t in checks:
        if not c:
            print(f"        FAIL: {t}")


def exists(p: str) -> bool:
    f = ROOT / p
    return f.is_file() and f.stat().st_size > 0


def png_ok(p: str) -> bool:
    f = ROOT / p
    return f.is_file() and f.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and f.stat().st_size > 20_000


def run_netlist(p: str) -> tuple[bool, str]:
    exe = find_ngspice(CFG.ngspice_path)
    text = (ROOT / p).read_text(encoding="utf-8")
    if CFG.spice_options not in text:
        return False, "mandatory .options line missing"
    with tempfile.TemporaryDirectory() as td:
        r = subprocess.run([str(exe), "-b", str(ROOT / p)], capture_output=True, text=True, timeout=120, cwd=td)
    out = r.stdout + r.stderr
    bad = [l for l in out.splitlines() if "error" in l.lower() or "singular" in l.lower()]
    printed = sum(1 for l in out.splitlines() if " = " in l and l.strip().split(" = ")[0].startswith(("i(", "v(")))
    return (r.returncode == 0 and not bad and printed > 0), f"exit {r.returncode}, {printed} values printed, errors: {bad[:2]}"


def main() -> int:
    # ---------------------------------------------------------------- 1 datasets
    ch = []
    for c in CASES:
        f = f"data/transistor/{c}.csv"
        if not exists(f):
            ch.append((False, f"{f} missing")); continue
        d = pd.read_csv(ROOT / f)
        ch += [(list(d.columns) == COLS, f"{c}: column order"), (len(d) == 10_000, f"{c}: {len(d)} rows"),
               (bool(np.isfinite(d.to_numpy(float)).all()), f"{c}: finite values")]
    item("1. Four transistor datasets (NMOS/PMOS ON/OFF), 10,000 rows, required columns",
         [f"data/transistor/{c}.csv" for c in CASES], ch)

    # ---------------------------------------------------------------- 2 netlists
    for label, files in (("2a. Example single-transistor netlists (4 cases)", [f"netlists/transistor/{c}.cir" for c in CASES]),
                         ("2b. Gate netlists (INV, NAND2, NOR2, AND2, OR2)",
                          [f"netlists/gates/{g}.cir" for g in ("INV", "NAND2", "NOR2", "AND2", "OR2")]),
                         ("2c. Test-circuit netlist (3-bit CLA, 192 transistors)", ["netlists/circuit/CLA3.cir"])):
        ch = []
        for f in files:
            if not exists(f):
                ch.append((False, f"{f} missing")); continue
            ok, why = run_netlist(f)
            ch.append((ok, f"{f}: runs in ngspice ({why})"))
        if "CLA3" in label and exists(files[0]):
            n_m = sum(1 for l in (ROOT / files[0]).read_text(encoding="utf-8").splitlines() if l[:1] in "Mm")
            ch.append((n_m == 192, f"CLA3 netlist has {n_m} MOSFETs"))
        if "single-transistor" in label:
            for f in files:
                t = (ROOT / f).read_text(encoding="utf-8").lower()
                ch.append((all(s in t for s in ("vd d 0", "vg g 0", "vs s 0", "vb b 0")), f"{f}: one source per terminal"))
        item(label, files, ch)

    # ---------------------------------------------------------------- 3 data check
    k = ROOT / "results/data_check/kcl_summary.csv"
    ch = [(k.is_file(), "kcl_summary.csv present")]
    if k.is_file():
        kk = pd.read_csv(k)
        ch += [(set(kk.case) == set(CASES), "KCL rows for all 4 cases"),
               (int(kk.n_over_tolerance.sum()) == 0, f"samples over KCL tolerance: {int(kk.n_over_tolerance.sum())}")]
    plots = ["results/data_check/fig_off_current_vs_temperature.png", "results/data_check/fig_off_current_vs_vth0.png",
             "results/data_check/fig_gate_current_vs_toxe.png"]
    ch += [(png_ok(p), f"{p} valid PNG") for p in plots]
    item("3. Data check: Kirchhoff per case + 3 trend plots", ["results/data_check/kcl_summary.csv", *plots,
                                                              "results/data_check/data_quality_report.md"], ch)

    # ---------------------------------------------------------------- 4 model comparison
    f = "results/a2/validation_comparison.csv"
    ch = [(exists(f), f"{f} present")]
    if exists(f):
        v = pd.read_csv(ROOT / f)
        ch += [(v.groupby("case").algo.nunique().min() >= 4 and v.case.nunique() == 4, "4 algorithms x 4 cases"),
               (all(c in v.columns for c in ("mean_r2_t", "mean_med_rel", "mean_p95_rel")), "R2, median and p95 columns")]
    item("4. Model comparison table: 4 model types x 4 cases (validation R2, median, p95)",
         [f, "results/a2/validation_comparison.md", "results/improve/experiments_validation.csv"], ch)

    # ---------------------------------------------------------------- 5 chosen models
    import joblib
    ch = []
    for c in CASES:
        f = ROOT / f"trained_models/{c}.joblib"
        try:
            m = joblib.load(f)
            p = m.predict(pd.read_csv(ROOT / f"data/final_eval/transistor_uniform/{c}.csv").head(5))
            ch.append((np.isfinite(p.to_numpy()).all(), f"{c}: model loads and predicts"))
        except Exception as e:  # noqa: BLE001
            ch.append((False, f"{c}: {e}"))
    t = ROOT / "results/final/transistor_test_results.csv"
    ch.append((t.is_file(), "final test-set table present"))
    if t.is_file():
        tt = pd.read_csv(t)
        ch.append((all(c in set(tt.case) for c in CASES) and {"med_rel", "p95_rel"} <= set(tt.columns),
                   "test median and p95 for every case"))
    figs = [f"results/final/fig_transistor_pred_vs_spice_{c}.png" for c in CASES]
    ch += [(png_ok(p), f"{p} valid PNG") for p in figs]
    item("5. Chosen models: test median / p95, predicted-vs-SPICE plot per case, 4 saved models",
         [f"trained_models/{c}.joblib" for c in CASES] + ["results/final/transistor_test_results.csv", *figs], ch)

    # ---------------------------------------------------------------- 6/7 gates
    g = ROOT / "results/final/gate_18cases_partB.csv"
    ch = [(g.is_file(), "18-case table present")]
    if g.is_file():
        gg = pd.read_csv(g, dtype={"inputs": str})
        ch += [(len(gg) == 18, f"{len(gg)} gate/input cases"),
               ({"spice_median_A", "ml_median_A", "bias", "median_err", "p95_err"} <= set(gg.columns),
                "SPICE median, ML median, bias, median error, p95 error")]
    r = pd.read_csv(ROOT / "data/gates/gate_spice_results.csv", dtype={"inputs": str})
    ch += [(len(r) == 3600 and (r.status == "ok").all() and r.logic_ok.all(), "3,600 gate SPICE references valid")]
    item("6. Gate results table: 18 gate/input cases", ["results/final/gate_18cases_partB.csv",
                                                        "results/final/gate_18cases_partB.md",
                                                        "data/gates/gate_spice_results.csv"], ch)
    item("7. Gate parity plot (log axes)", ["results/final/fig_gate_parity.png"],
         [(png_ok("results/final/fig_gate_parity.png"), "valid PNG")])

    # ---------------------------------------------------------------- 8 test circuit
    c = ROOT / "results/final/cla3_results.csv"
    halves = [ROOT / "data/circuit/cla3_spice_results.csv", ROOT / "data/circuit/cla3_spice_results_cin1.csv"]
    s = pd.concat([pd.read_csv(f) for f in halves if f.is_file()], ignore_index=True)
    combos = s[["vector_id", "corner_id"]].drop_duplicates()
    ch = [(c.is_file(), "evaluation table present"), (png_ok("results/final/fig_cla3_parity.png"), "parity plot valid PNG"),
          (len(s) == 25600 and len(combos) == 25600 and s.vector_id.nunique() == 128 and set(s.cin) == {0, 1},
           "every input combination: 128 vectors (A, B, Cin) x 200 corners = 25,600 CLA SPICE references"),
          ((s.status == "ok").all() and s.sum_ok.all() and s.all_nodes_logic_ok.all(),
           "all converged, A+B+Cin correct, every internal net at its logic level")]
    if c.is_file():
        cc = pd.read_csv(c)
        ch.append(({"bias", "median_err", "p95_err", "spice_median_A", "ml_median_A"} <= set(cc.columns), "metrics present"))
        ch.append(((cc.get("vectors") == 128).all(), "evaluation covers all 128 input combinations"))
    item("8. Test-circuit results (same metrics) + parity plot", ["results/final/cla3_results.csv",
                                                                 "results/final/fig_cla3_parity.png",
                                                                 "data/circuit/cla3_spice_results.csv",
                                                                 "data/circuit/cla3_spice_results_cin1.csv"], ch)

    # ---------------------------------------------------------------- 9 code
    scripts = ["run_all.py", "config/project.toml", "src/leakage/simulator.py", "src/leakage/transistor.py",
               "src/leakage/dataset.py", "src/leakage/ml.py", "src/leakage/gates.py", "src/leakage/gate_ml.py",
               "src/leakage/circuit.py", "scripts/a1_generate_datasets.py", "scripts/a2_train_models.py",
               "scripts/b3_gate_spice.py", "scripts/b4_gate_ml.py", "scripts/c1_cla_spice.py", "scripts/c2_cla_ml.py"]
    ch = [(exists(p), f"{p} present") for p in scripts]
    ch.append((all(k in CFG.seeds for k in ("data", "split", "training", "corners")), "fixed seeds in config/project.toml"))
    item("9. Code that regenerates everything, run in order, fixed seeds", scripts, ch)

    # ---------------------------------------------------------------- 10 report + reproducibility docs
    rep = ROOT / "reports/FINAL_REPORT.md"
    ch = [(rep.is_file(), "reports/FINAL_REPORT.md present")]
    if rep.is_file():
        t = rep.read_text(encoding="utf-8").lower()
        for kw in ("setup", "data generation", "models", "gate results", "test-circuit", "error analysis", "limitations"):
            ch.append((kw in t, f"report covers '{kw}'"))
    ch += [(exists("reports/FINAL_REPORT.html"), "HTML version present"), (exists("README.md"), "README"),
           (exists("docs/BEGINNER_GUIDE.md"), "Linux install guide"), (exists("requirements.txt"), "requirements.txt")]
    item("10. Typed report + reproducibility documentation",
         ["reports/FINAL_REPORT.md", "reports/FINAL_REPORT.html", "README.md", "docs/BEGINNER_GUIDE.md", "requirements.txt"], ch)

    # ---------------------------------------------------------------- write checklist
    lines = ["# Deliverables checklist (generated by scripts/z2_audit.py)", "",
             "Every item is verified by content (see the checks column). Paths are relative to the project root.", "",
             "| # | Requirement (project PDF, Section 6) | Status | Output files | Checks |", "|---|---|---|---|---|"]
    for k, it in enumerate(ITEMS, 1):
        files = "<br>".join(f"`{f}`" for f in it["files"])
        checks = "<br>".join(("✔ " if c else "✘ ") + t for c, t in it["checks"])
        lines.append(f"| {k} | {it['req']} | {'✅ verified' if it['ok'] else '❌ FAILED'} | {files} | {checks} |")
    n_ok = sum(it["ok"] for it in ITEMS)
    lines += ["", f"**{n_ok}/{len(ITEMS)} deliverable groups verified.**", ""]
    (ROOT / "reports/deliverables_checklist.md").write_text("\n".join(lines), encoding="utf-8")
    (ROOT / "results/final/audit.json").write_text(json.dumps([{**i, "checks": [[bool(c), t] for c, t in i["checks"]]}
                                                              for i in ITEMS], indent=1), encoding="utf-8")
    print(f"\n{n_ok}/{len(ITEMS)} deliverable groups verified -> reports/deliverables_checklist.md")
    return 0 if n_ok == len(ITEMS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
