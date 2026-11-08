# Installation guide (Windows 10/11)

Tested on **Windows 11 Pro with Python 3.13.14 and ngspice 44.2**. Every command below is PowerShell, run from the project root (the folder that contains `run_all.py`). No Docker, WSL or Linux paths are needed.

## 1. Prerequisites

| Tool | Version tested | Notes |
|---|---|---|
| Python | 3.13.14 (needs ≥ 3.11) | python.org installer or the Microsoft Store build; `python --version` must work |
| Git | any recent | only to clone the repository |
| ngspice | 44.2 (64-bit) | ngspice.sourceforge.io → *Download* → *ngspice-44.2_64.7z* (MS Windows). It is not a pip package. |

## 2. Get the project

```powershell
git clone https://github.com/realsvira/DVD_Project.git
cd DVD_Project
```

## 3. Install ngspice and tell the project where it is

1. Extract the archive to any folder, for example `C:\Spice64`.
2. In a terminal, use **`ngspice_con.exe`**. `ngspice.exe` is the GUI build and opens its own window.
3. Check it:
   ```powershell
   & "C:\Spice64\bin\ngspice_con.exe" -v
   ```
4. The project looks for ngspice in this order:
   1. the `NGSPICE` environment variable;
   2. `[ngspice].path` in `config/project.toml`;
   3. `ngspice_con` or `ngspice` on PATH;
   4. common folders (`C:\Spice64\bin`, `C:\Program Files\Spice64\bin`, ...).

   The simplest option is to set the variable, either for the current terminal:
   ```powershell
   $env:NGSPICE = "C:\Spice64\bin\ngspice_con.exe"
   ```
   or permanently, for your user (open a new terminal afterwards):
   ```powershell
   [Environment]::SetEnvironmentVariable("NGSPICE", "C:\Spice64\bin\ngspice_con.exe", "User")
   ```
   Alternatively, edit `path` under `[ngspice]` in `config/project.toml`.

## 4. Python environment

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
To *re-train* the final transistor models you also need PyTorch, CPU build, about 200 MB. The saved models predict without it.
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-train.txt
```
To activate the venv instead of typing the full path:
```powershell
.\.venv\Scripts\Activate.ps1
```
If activation is blocked, run this once:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## 5. Verify the installation

```powershell
.\.venv\Scripts\python.exe run_all.py --mode verify
```
Success looks like this:
- the environment check ends with `ENVIRONMENT OK`;
- `pytest` reports all tests passed;
- the audit prints `12/12 deliverable groups verified`.

Logs are written to `results\run_logs\`.

## 6. Run the project

| Command | What it does | Time |
|---|---|---|
| `python run_all.py --list` | lists every stage of every mode | – |
| `python run_all.py --mode smoke` | small end-to-end check of every stage (writes only to `data\transistor_small`, `results\a2_quick` and smoke files) | ~10 min |
| `python run_all.py --mode full` | regenerates everything in the original order | many hours (training dominates) |
| `python run_all.py --mode full --from b3` | resumes from a stage | – |
| `python run_all.py --mode full --only c1 c2` | runs only selected stages | – |

In each command, `python` means `.\.venv\Scripts\python.exe`, or plain `python` once the venv is activated.

Notes:
- The generation stages are resumable (saved shards), and the training stages cache every fitted configuration.
- The three one-time evaluations (A2 test, A2b fresh test, improvement final evaluation) refuse to run a second time. The runner reports them as *skipped (already evaluated)*.
- The final transistor models (`trained_models\*.joblib`) are part of the repository, so all prediction and evaluation stages run without re-training.

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `ngspice` "not recognized" | Use the full path to `ngspice_con.exe`, or set `NGSPICE` (step 3). |
| `ENVIRONMENT NOT READY`: ngspice not found | Same as above; `check_environment.py` prints where it looked. |
| `ModuleNotFoundError` | Use the venv's Python (`.\.venv\Scripts\python.exe`) and reinstall the requirements. |
| Garbled characters in console output | Run `$env:PYTHONIOENCODING = "utf-8"`; the runner sets it for child processes. |
| A one-time evaluation says "refusing to evaluate again" | Expected: those results already exist. Use `--force` only to deliberately repeat it. |
