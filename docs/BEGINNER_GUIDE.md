# Beginner guide: running this project from zero

This guide assumes you have a Windows 10/11 PC and have never used Python, Git or SPICE. Follow the steps in order. Every command is typed into **PowerShell**: press the Windows key, type `PowerShell`, press Enter. Paste a command with right-click, then press Enter.

---

## Step 1: Install Python

1. Go to **python.org → Downloads → Windows** and download **Python 3.13** (any 3.11 or newer works).
2. Run the installer. **Tick "Add python.exe to PATH"** at the bottom of the first screen, then click *Install Now*.
3. Close PowerShell, open it again and check:
   ```powershell
   python --version
   ```
   You should see something like `Python 3.13.x`.

## Step 2: Install Git and download the project

1. Go to **git-scm.com → Download for Windows**, install with all default options, and reopen PowerShell.
2. Download the project into your user folder:
   ```powershell
   cd $HOME
   ```
   ```powershell
   git clone https://github.com/realsvira/DVD_Project.git
   ```
   ```powershell
   cd DVD_Project
   ```
   *(If you were given a ZIP file instead, right-click it → Extract All, then use `cd` to go into the extracted `DVD_Project` folder.)*

From now on, **always run commands inside the `DVD_Project` folder**. If you open a new PowerShell window, first run `cd $HOME\DVD_Project`.

## Step 3: Install ngspice (the circuit simulator)

1. Go to **ngspice.sourceforge.io → Download** and get the **MS Windows 64-bit** archive (version 44.2 was used).
2. Extract it so that this file exists: `C:\Spice64\bin\ngspice_con.exe`. If the archive is `.7z`, use 7-Zip to extract it.
3. Tell the project where ngspice is. This is permanent; reopen PowerShell afterwards:
   ```powershell
   [Environment]::SetEnvironmentVariable("NGSPICE", "C:\Spice64\bin\ngspice_con.exe", "User")
   ```
4. Check it:
   ```powershell
   & $env:NGSPICE -v
   ```
   It should print `ngspice-44.2` (or similar).

## Step 4: Create the Python environment (one time, about 5 minutes)

```powershell
python -m venv .venv
```
```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
This downloads the libraries the project uses (numpy, pandas, scikit-learn, matplotlib and so on) into the `.venv` folder, so nothing else on your PC is affected.

## Step 5: Check that everything works (about 1 minute)

```powershell
.\.venv\Scripts\python.exe run_all.py --mode verify
```
At the end you should see a table where every stage says **ok**:
```
| env   | ok |
| tests | ok |
| audit | ok |
```
- **env:** Python, the libraries, ngspice and the transistor model file were all found.
- **tests:** all 90 automatic tests passed.
- **audit:** all 12 project deliverables are present and correct.

## Step 6: Run a quick demonstration (about 1–2 minutes)

```powershell
.\.venv\Scripts\python.exe run_all.py --mode smoke
```
This runs a small version of every stage:
- simulates transistors, gates and the 3-bit adder in ngspice;
- trains a tiny ML model;
- predicts a transistor's currents with the final model and compares them with SPICE.

Every stage should say **ok**.

## Step 7: Try the trained model yourself

Predict the four currents of one transistor and compare with a real SPICE simulation:
```powershell
.\.venv\Scripts\python.exe scripts\a2_predict.py --device nmos --vd 1.0 --vg 0 --vs 0 --vb 0 --vdd 1.0 --temp 27 --w 90e-9 --spice
```
Change the numbers to try other conditions:

| Option | Meaning |
|---|---|
| `--device` | `nmos` or `pmos` |
| `--vd`, `--vg`, `--vs`, `--vb` | drain, gate, source and body voltages (V) |
| `--vdd` | supply voltage, 0.9–1.1 V |
| `--temp` | temperature, −40 to 125 °C |
| `--w` | transistor width, 45e-9 to 360e-9 m |

Rules:
- The gate must be exactly 0 or VDD.
- The NMOS body must be 0, and the PMOS body must equal VDD.

## Step 8: Look at the results

| What | Where |
|---|---|
| **Final report** (read this first) | `reports\FINAL_REPORT.html`: double-click to open it in a browser. To get a PDF, press Ctrl+P and choose "Save as PDF". |
| Which file answers which project requirement | `reports\deliverables_checklist.md` |
| Final plots and tables | `results\final\` |
| Datasets | `data\transistor\` |
| SPICE netlists | `netlists\` |
| Trained models | `trained_models\` |

To open a folder in File Explorer from PowerShell, use for example `explorer results\final`.

## Step 9 (optional, advanced): regenerate everything from scratch

```powershell
.\.venv\Scripts\python.exe run_all.py --list
```
That shows every stage.
```powershell
.\.venv\Scripts\python.exe run_all.py --mode full
```
- **Warning:** this takes many hours. Training the neural networks alone is roughly 15 hours on a 12-core PC.
- Re-training the final models also needs PyTorch:
  ```powershell
  .\.venv\Scripts\python.exe -m pip install -r requirements-train.txt
  ```
- If it stops, continue with `--from <stage name>`.
- Results that were already evaluated once are protected and will be reported as "skipped".

---

## If something goes wrong

| Message | What to do |
|---|---|
| `python` is not recognized | Python was installed without "Add to PATH". Reinstall Python and tick that box. |
| `ngspice not found` / `ENVIRONMENT NOT READY` | Redo Step 3; make sure `C:\Spice64\bin\ngspice_con.exe` exists and reopen PowerShell. |
| `No module named ...` | You used plain `python` instead of `.\.venv\Scripts\python.exe`, or Step 4 did not finish; run Step 4 again. |
| `cannot be loaded because running scripts is disabled` | Only happens if you try `Activate.ps1`. You don't need it: use `.\.venv\Scripts\python.exe` as shown. |
| A stage says FAILED | Open the log file it names (in `results\run_logs\`) and read the last lines. |

More detail: `docs\INSTALL_WINDOWS.md` and `README.md`.
