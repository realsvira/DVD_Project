# ML-Driven PVT Variation Aware Leakage Estimation

Digital VLSI Design course project. Specification: `DVD_ML_PROJECT.pdf` (the primary reference).

| Item | Team value |
|---|---|
| Model file | **45nm HP**: PTM 45nm High Performance, Metal Gate / High-K / Strained-Si (`models/45nm_HP.pm`) |
| Nominal VDD | **1.0 V** (sampled 0.9 to 1.1 V) |
| Channel length | **L = 45 nm**; widths W = 45 to 360 nm |
| Test circuit | **3-bit carry-lookahead adder** |
| Simulator | ngspice 44.2 (console build `ngspice_con.exe`) |
| Language | Python 3.11+ (developed on 3.13, Windows 11) |

Team: Saumya Vira, Anish Toshniwal, Shrenil Patel, Het Selarka.

---

## Project layout

```
Project/
├─ DVD_ML_PROJECT.pdf, Project-1_*.pdf,     <- supplied inputs, kept untouched
│  22n_NMOS_ON.net, *.zip, model/, models/
├─ config/project.toml      <- central configuration (assignment, ranges, seeds, paths)
├─ src/leakage/             <- Python package
│   ├─ config.py            <- loads project.toml into a typed Config object
│   └─ spice.py             <- model-card parsing/editing, ngspice locate/run/parse
├─ scripts/
│   ├─ setup_env.ps1        <- one-shot setup (venv + deps + checks)
│   ├─ inventory_inputs.py  <- extracts every archive, hashes, compares model copies
│   └─ check_environment.py <- verifies Python, packages, model file, ngspice
├─ tests/                   <- pytest unit tests (config + SPICE helpers)
├─ netlists/{transistor,gates,circuit}/   <- example SPICE netlists (deliverable 2)
├─ data/{transistor,gates,circuit}/       <- generated datasets (CSV)
├─ trained_models/          <- saved ML models (NOT the SPICE .pm model cards)
├─ results/  figures/       <- tables, metrics, plots
├─ reports/                 <- input_inventory.md, environment_check.txt, final report
└─ inputs/extracted/        <- regenerated extraction of the zip/tar archives
```

A note on the word "model": `models/*.pm` are **SPICE transistor model cards** (BSIM4).
They are not ML models. Trained ML models go in `trained_models/`.

---

## Setup (Windows)

### 1. Check Python
PowerShell:
```powershell
python --version
```
You need 3.11 or newer.

### 2. Find ngspice
`ngspice -v` gives "not recognized" because ngspice is not on PATH. On this machine it is at:
```
D:\VLSI\NGSPICE\ngspice-44.2_64\Spice64\bin\ngspice_con.exe
```
Use **`ngspice_con.exe`** in a terminal. `ngspice.exe` is the GUI build and opens its own window.

To locate it on another machine:
```powershell
Get-ChildItem C:\,D:\ -Recurse -Filter ngspice_con.exe -ErrorAction SilentlyContinue | Select-Object -First 5 FullName
```
To verify it:
```powershell
& "D:\VLSI\NGSPICE\ngspice-44.2_64\Spice64\bin\ngspice_con.exe" -v
```

The project does **not** need ngspice on PATH. It looks in this order:
1. the `NGSPICE` environment variable,
2. `[ngspice].path` in `config/project.toml`,
3. `ngspice_con` / `ngspice` on PATH,
4. common install folders (`C:\Spice64\bin`, ...).

Teammates with a different install location can set the path for one session:
```powershell
$env:NGSPICE = "C:\Spice64\bin\ngspice_con.exe"
```
Optional: to add it permanently to your **user** PATH (open a new terminal afterwards):
```powershell
[Environment]::SetEnvironmentVariable("Path", [Environment]::GetEnvironmentVariable("Path","User") + ";D:\VLSI\NGSPICE\ngspice-44.2_64\Spice64\bin", "User")
```

### 3. Create the virtual environment and install dependencies
From the project root (`D:\DVD\Project`):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If activation is blocked by the execution policy, run this once:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```
Alternatively, skip activation and call `.\.venv\Scripts\python.exe` directly.

CMD equivalent:
```bat
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

Or do all of setup in one step:
```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
```

### 4. Verify
```powershell
.\.venv\Scripts\python.exe scripts\inventory_inputs.py
.\.venv\Scripts\python.exe scripts\check_environment.py
.\.venv\Scripts\python.exe -m pytest -q tests
```
Setup succeeded when:
* `check_environment.py` ends with **`ENVIRONMENT OK`** and exits with code 0 (every line is `[PASS]`), and
* pytest reports all tests passed.

`check_environment.py` confirms that:
- Python is new enough and all packages import;
- the config loads;
- `models/45nm_HP.pm` exists and its SHA-256 matches the config;
- both model cards are BSIM4 (level 54) and the nominal vth0/toxe/ndep/u0/xj values parse;
- ngspice is found and answers `-v`;
- Python can run an ngspice batch simulation of NMOS OFF and PMOS OFF using the required `.options`;
- the simulator loaded exactly the file's parameter values (via `showmod`);
- the four terminal currents of each device sum to ~0 (Kirchhoff's current law);
- editing a model card (toxe −10 %) raises gate leakage as expected.

Results are also written to `reports/environment_check.txt`.

---

## Conventions and design decisions

* **Current sign:** every terminal current is positive when it flows **into** the device.
  ngspice reports `i(Vx)` flowing + to − through the source, so I_terminal = −i(Vx).
* **Units (SI, as in the BSIM4 cards):** W, L, toxe, xj in m; vth0 in V; ndep in cm⁻³;
  u0 in m²/(V·s); temperature in °C; currents in A.
* **Process variation is written into a fresh model card per sample.**
  Do not use `altermod` for `toxe`. ngspice 44.2 accepts the new value but does not recompute the
  oxide-derived terms, so gate and channel currents come out wrong by up to ~5×. This was measured
  in this project against fresh netlists. vth0, ndep, u0 and xj behaved correctly under `altermod`,
  but the project uses one trusted code path for all five parameters.
* **Only `toxe` is varied.** toxp, toxm and toxref keep their nominal values, as the spec lists only toxe.
* **Model copy:** `models/45nm_HP.pm` (from `Model Files-20261112.zip`). The older copies in `model/`
  and `model.tar` differ only in whitespace. See `reports/input_inventory.md`.
* **Simulator note:** ngspice reports the `version = 4.0` BSIM4 cards as "BSIM4v5". The printed
  "No compatibility mode selected!" is informational only.
* **Reproducibility:** all random seeds live in `config/project.toml` under `[seeds]`.

## Part A1: transistor simulation engine

### Modules
| File | Role |
|---|---|
| `src/leakage/simulator.py` | `NgspiceSimulator.run_op(circuits, vectors)`. A general batch DC operating-point engine, reused later for gates and the CLA. |
| `src/leakage/transistor.py` | Single-FET netlist generator and API: `make_inputs`, `sample_inputs`, `transistor_circuit`, `example_netlist`, `simulate_transistors`, `simulate_transistor`, `kcl_residual`, `kcl_ok`. |
| `src/leakage/spice.py` | Model-card read/edit, ngspice discovery, ASCII raw-file parser (`parse_raw_ascii`). |
| `scripts/a1_smoke_test.py` | Four-case nominal smoke test plus a 192-point corner sweep. Also writes the example netlists. |
| `netlists/transistor/*.cir` | Stand-alone example netlist for each case (deliverable 2). |

### How a simulation runs
1. Each sample becomes a self-contained circuit containing:
   - the required `.options` line;
   - `.temp`;
   - one FET with `L = 45 nm`;
   - a separate voltage source on each of d, g, s and b;
   - a copy of that device's own `.model` card from `45nm_HP.pm`, with the sample's vth0/toxe/ndep/u0/xj written in.

   The other device's card is left out, and the supplied `.pm` file is only ever read (a test checks its SHA-256).
2. About 200 circuits share one ngspice process. A driver runs `source s<k>.cir` → `op` →
   `write o<k>.raw i(vd) i(vg) i(vs) i(vb)` for each one, and up to 11 processes run in parallel.
3. Values come from ngspice's **ASCII raw files** (structured, full double precision), never from the printed log.
4. A circuit that fails writes no raw file. It is marked `failed` and re-run on its own to collect
   its exact error messages. Other circuits are unaffected (tested with an unknown-model circuit and a singular-matrix circuit).
5. Results are identical (relative difference < 1e-9) to running each example netlist on its own (tested).

### Sign convention
Every current is **positive when it flows into the device terminal**: `I_X = -i(VX)`.
At the nominal NMOS ON point, `i(vd) = -1.1208e-04`, so Id = +1.1208e-04 A, flowing into the drain.

### Kirchhoff check
The pass criterion is |Id + Ig + Is + Ib| ≤ 1e-6·max|I| + 10 fA.

At the nominal points the residual is 1e-17 A or less. When every current is tiny (≤ 1e-12 A, e.g. Vd = Vs), ngspice's BSIM4 code leaves a fixed residual of about 1.2 fA. That figure did not change with gmin, reltol or vntol (measured), so it comes from the device model, not from solver tolerances.

### How this differs from the supplied `22n_NMOS_ON.net`
That file was used for reference only. Its Linux `.include` path, `GMIN=1e-20` options, 22 nm sizes, Vg-referenced-to-`Ref` VDD sweep and `dc TEMP` sweep were all replaced:
- every sample gets one `op`;
- VDD, T, W, Vd, Vs and the process parameters are independent inputs;
- the required options line is used;
- the model card is embedded, so the netlist contains no paths.

### Reproduce
```powershell
.\.venv\Scripts\python.exe -m pytest -q tests
.\.venv\Scripts\python.exe scripts\a1_smoke_test.py
& "D:\VLSI\NGSPICE\ngspice-44.2_64\Spice64\bin\ngspice_con.exe" -b netlists\transistor\nmos_on.cir
```
The smoke-test results go to `results/a1_smoke_test.csv`, `results/a1_corner_sweep.csv` and `reports/a1_smoke_test.md`.

### API example
```python
from leakage.config import load_config
from leakage.transistor import simulate_transistor, sample_inputs, simulate_transistors
cfg = load_config()
r  = simulate_transistor(cfg, "nmos_off", vdd=0.95, temp=85, w=180e-9, vd=0.95, vs=0.0,
                         process={"vth0": 0.45, "toxe": 1.2e-9})
df = simulate_transistors(cfg, "pmos_on", sample_inputs(cfg, "pmos_on", 100))
```

## Part A1: dataset generation

Script: `scripts/a1_generate_datasets.py`; logic in `src/leakage/dataset.py`.

```powershell
# 20-sample smoke test per case (run after any code change)
.\.venv\Scripts\python.exe scripts\a1_generate_datasets.py --smoke
# quick end-to-end run: 200 samples per case into data\transistor_small\
.\.venv\Scripts\python.exe scripts\a1_generate_datasets.py --small
# full run: 10,000 valid samples per case into data\transistor\ (smoke test runs first)
.\.venv\Scripts\python.exe scripts\a1_generate_datasets.py
```
Other options: `--cases nmos_off pmos_off`, `--n 2000`, `--out data\tmp`, `--batch 500`,
`--workers 6`, `--max-extra 200`, `--skip-smoke`, `--no-progress`.

**Sampling** (spec Section 4):
- VDD ~ U(0.9, 1.1) V, drawn first.
- T ~ U(−40, 125) °C.
- W ~ U(45, 360) nm.
- Vd and Vs: each drawn independently from U(0, VDD).
- vth0/toxe/ndep/u0/xj: each drawn independently from U(±10 %) around that device's own nominal value.
- Vg and Vb are set by the case.

**Reproducibility and stable IDs:**
- Sample *k* of a case is drawn from `default_rng([case_seed, k // 1000])`, and every block of 1000 is always drawn in full.
- So sample *k* gets identical inputs whatever `--n` is, and the small run is an exact prefix of the full run.
- Per-case seeds are `seeds.data + 100·case_index` (1001, 1101, 1201, 1301).
- Inputs are rounded to the 12 significant digits written into the netlist, so the CSV holds exactly what was simulated.

**Robustness:**
- Each batch (`--batch`, default 1000) is saved once, atomically, to `data/transistor/_shards/<case>/`.
- Rerunning the same command resumes: saved IDs are skipped, so nothing is duplicated (tested by deleting shards). Ctrl+C is safe.
- Failed simulations keep NaN currents, never zeros. They are listed in `failures.csv`, and the next unused IDs are simulated until there are N valid samples. The run stops if more than `--max-extra` replacements would be needed.
- Parallelism is a fixed pool of `--workers` ngspice processes, 11 by default.

**Outputs** in `data/transistor/`:

| File | Content |
|---|---|
| `<case>.csv` | the dataset: columns `vdd, temp, w, vd, vs, vg, vb, vth0, toxe, ndep, u0, xj, id, ig, is, ib`, sorted by sample ID |
| `<case>_index.csv` | maps each row to its sample ID, plus that row's Kirchhoff sum |
| `failures.csv` | each failed simulation with its inputs and simulator messages |
| `metadata.json` | seeds, ranges, versions, git commit, model hash, timings, validation results, current statistics |
| `summary.md` | per-case counts and checks (also copied to `reports/a1_dataset_summary.md`) |

**Validation** runs on every final CSV and checks:
- the row count, column order and that every value is finite;
- every input range, and the Vg/Vb case rule;
- each process parameter is within ±10 % of the correct device's nominal;
- input rows and sample IDs are unique, and no current is exactly zero;
- Kirchhoff's law holds for every row.

## Part A1: data check (deliverable 3)

```powershell
.\.venv\Scripts\python.exe scripts\a1_data_check.py
```
This takes about 2 minutes. It reads `data/transistor/` and writes everything to `results/data_check/`:

| File | Content |
|---|---|
| `data_quality_report.md` | written report (also copied to `reports/a1_data_quality_report.md`) |
| `kcl_summary.csv` | Id+Ig+Is+Ib for each case: max / median / p95, absolute and relative |
| `quality_checks.csv` | range, missing-value, uniqueness and failure checks |
| `fig_off_current_vs_temperature.png`, `fig_off_current_vs_vth0.png`, `fig_gate_current_vs_toxe.png` | the three required trend plots (NMOS and PMOS panels) |
| `fig_supp_off_current_vs_source_offset.png`, `source_offset_effect.csv` | why the OFF-current plots condition on the source potential |
| `trend_slopes.csv` | slopes with the other inputs held fixed vs controlled SPICE sweeps |
| `controlled_sweeps.csv` | reference SPICE sweeps varying one input at a time (not training data) |
| `suspicious_samples.csv` | flagged samples with the reason (nothing is removed) |
| `warning_audit.csv` | flagged plus random samples re-simulated alone: exact messages and a reproduction check |

**Key findings:**
- **Kirchhoff:** 0 of 40,000 samples exceed the tolerance; the largest absolute residual is about 2e-15 A.
- **Trends:** all three have the expected sign, and their slopes match the controlled sweeps (for example, NMOS Ig vs toxe: −0.66 vs −0.65 decades per 0.1 nm).
- **Conditioning:** the OFF-current plots must condition on the source potential, not just |Vds|. Each 0.1·VDD that the source sits off its rail cuts the current by about 1 decade.
- **toxe warning:** every sample with a non-nominal toxe triggers the BSIM4 warning "toxe != toxp + dtox; dtox ignored", because the spec varies toxe only. It is documented as a modelling assumption.

## Workflow status

| Stage | Status |
|---|---|
| 0. Setup, inventory, environment check | done |
| A1. Simulation engine, example netlists, smoke test | done |
| A1. 4 × 10,000 datasets | done |
| A1. Data check (Kirchhoff summary, 3 trend plots, quality report) | done |
| A2. ML model comparison and selection | next |
| A2. ML model comparison and selection | - |
| B. Gate-level validation (18 cases × 200 corners) | - |
| C. 3-bit CLA validation | - |
| Report | - |
