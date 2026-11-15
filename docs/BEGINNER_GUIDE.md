# Beginner guide: running this project from zero (Linux)

This guide assumes a Linux PC (Fedora, Ubuntu/Debian or Arch) and no experience with Python, Git or SPICE. Follow the steps in order. Every command is typed into a **terminal**: open it from your applications menu (search for "Terminal"; on Ubuntu, Ctrl+Alt+T also works). To paste a command, use Ctrl+Shift+V, then press Enter.

Commands starting with `sudo` install software for the whole system and ask for your password. Nothing is shown while you type it; this is normal.

---

## Step 1: Install Python, Git and ngspice (the circuit simulator)

Run the line for your distribution:

| Distribution | Command |
|---|---|
| Fedora | `sudo dnf install python3 git ngspice` |
| Ubuntu / Debian | `sudo apt update && sudo apt install python3 python3-venv git ngspice` |
| Arch | `sudo pacman -S python git ngspice` |

Then check the versions:
```bash
python3 --version
```
You need `Python 3.11` or newer.
```bash
ngspice -v
```
You should see `ngspice-44.2` or newer. Version 45.2 was used to verify this guide.

*(If Python is older than 3.11, as on Ubuntu 22.04, install a newer Python first. One easy way is `curl -LsSf https://astral.sh/uv/install.sh | sh` and then `uv python install 3.13`. In the later steps, use `uv venv --python 3.13 .venv` instead of `python3 -m venv .venv`.)*

## Step 2: Download the project

```bash
cd ~
```
```bash
git clone https://github.com/realsvira/DVD_Project.git
```
```bash
cd DVD_Project
```
*(If you were given a ZIP file instead, extract it with `unzip DVD_Project.zip`, then use `cd` to go into the extracted `DVD_Project` folder.)*

From now on, **always run commands inside the `DVD_Project` folder**. If you open a new terminal, first run `cd ~/DVD_Project`.

## Step 3: Create the Python environment (one time, about 5 minutes)

```bash
python3 -m venv .venv
```
```bash
.venv/bin/python -m pip install --upgrade pip
```
```bash
.venv/bin/python -m pip install -r requirements.txt
```
This downloads the libraries the project uses (numpy, pandas, scikit-learn, LightGBM, matplotlib and so on) into the `.venv` folder, so nothing else on your PC is affected.

*Optional:* run `source .venv/bin/activate` to type `python` instead of `.venv/bin/python` in that terminal. This guide always uses the full form, so you don't need it.

## Step 4: Check that everything works (about 1 minute)

```bash
.venv/bin/python run_all.py --mode verify
```
At the end you should see a table where every stage says **ok**:
```
| env   | ok |
| tests | ok |
| audit | ok |
```
- **env:** Python, the libraries, ngspice and the transistor model file were all found.
- **tests:** all 92 automatic tests passed.
- **audit:** all 12 project deliverables are present and correct.

## Step 5: Run a quick demonstration (a few minutes)

```bash
.venv/bin/python run_all.py --mode smoke
```
This runs a small version of every stage:
- simulates transistors, gates and the 3-bit adder in ngspice;
- trains a tiny ML model;
- predicts a transistor's currents with the final model and compares them with SPICE.

Every stage should say **ok**.

## Step 6: Try the trained model yourself

Predict the four currents of one transistor and compare with a real SPICE simulation:
```bash
.venv/bin/python scripts/a2_predict.py --device nmos --vd 1.0 --vg 0 --vs 0 --vb 0 --vdd 1.0 --temp 27 --w 90e-9 --spice
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

## Step 7: Look at the results

| What | Where |
|---|---|
| **Final report** (read this first) | `reports/FINAL_REPORT.pdf`, or `reports/FINAL_REPORT.html` in a browser |
| Which file answers which project requirement | `reports/deliverables_checklist.md` |
| Final plots and tables | `results/final/` |
| Datasets | `data/transistor/` |
| SPICE netlists | `netlists/` |
| Trained models | `trained_models/` |

To open a file or folder from the terminal, use `xdg-open`, for example `xdg-open reports/FINAL_REPORT.html` or `xdg-open results/final`.

## Step 8 (optional, advanced): regenerate everything from scratch

```bash
.venv/bin/python run_all.py --list
```
That shows every stage.
```bash
.venv/bin/python run_all.py --mode full
```
- **Warning:** this takes many hours. Training the neural networks alone is roughly 15 hours on a 12-core PC.
- Re-training the final models also needs PyTorch (CPU build):
  ```bash
  .venv/bin/python -m pip install -r requirements-train.txt
  ```
- If it stops, continue with `--from <stage name>`.
- Results that were already evaluated once are protected and will be reported as "skipped".
- To keep it running after you close the terminal, start it with `nohup .venv/bin/python run_all.py --mode full &`.

---

## If something goes wrong

| Message | What to do |
|---|---|
| `python3: command not found` | Redo Step 1. |
| `ensurepip is not available` when creating `.venv` | Ubuntu/Debian: `sudo apt install python3-venv`, then `rm -rf .venv` and redo Step 3. |
| `ngspice: command not found` / `ENVIRONMENT NOT READY` | Redo Step 1. If ngspice is installed somewhere unusual, run `export NGSPICE=/full/path/to/ngspice` first. |
| `[FAIL] Model file SHA-256 matches config` | Your copy's line endings were changed. Delete the `DVD_Project` folder and redo Step 2. |
| `No module named ...` | You used `python3` instead of `.venv/bin/python`, or Step 3 did not finish; run Step 3 again. |
| `Permission denied` | You are outside your home folder. Run `cd ~/DVD_Project`. Never use `sudo` for the project's own commands. |
| A stage says FAILED | Open the log file it names (in `results/run_logs/`) and read the last lines. |

More detail: the *Installation (Linux)* section of `README.md`.
