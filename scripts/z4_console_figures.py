"""Console screenshots for the LaTeX report: real tool output rendered as terminal-style images.

Every image shows actual project output (no text is typed by hand):
  screen_netlist.png      first lines of netlists/transistor/nmos_off.cir
  screen_ngspice.png      ngspice run of netlists/gates/NAND2.cir (live run)
  screen_dataset.png      first rows of data/transistor/nmos_off.csv
  screen_nodes.png        node-voltage columns of results/final/gate_predictions_partB.csv
  screen_cla_cin1.png     Cin = 1 full-circuit SPICE run log (results/c1_cin1_log.txt)
  screen_verify.png       run_all.py --mode verify summary (results/run_logs/verify_summary.md)

Output: reports/figures/screens/*.png
Run:  .\\.venv\\Scripts\\python.exe scripts\\z4_console_figures.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from leakage.config import load_config  # noqa: E402
from leakage.spice import find_ngspice  # noqa: E402

CFG = load_config()
OUT = CFG.reports / "figures" / "screens"
BG, FG, PROMPT = "#1e1e1e", "#d4d4d4", "#6a9955"


def render(name: str, command: str, lines: list[str], width_chars: int = 104) -> None:
    lines = [ln.rstrip()[:width_chars] for ln in lines]
    h = 0.62 + 0.205 * (len(lines) + 1)
    fig = plt.figure(figsize=(0.0835 * width_chars + 0.4, h), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off(); ax.set_facecolor(BG)
    y = 1 - 0.32 / h
    ax.text(0.012, y, "PS DVD_Project> " + command, color=PROMPT, family="monospace", fontsize=9.2,
            va="top", transform=ax.transAxes)
    for ln in lines:
        y -= 0.205 / h
        ax.text(0.012, y, ln, color=FG, family="monospace", fontsize=9.2, va="top", transform=ax.transAxes)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200, facecolor=BG); plt.close(fig)
    print("wrote", (OUT / name).relative_to(ROOT))


def main() -> int:
    net = (CFG.netlists / "transistor" / "nmos_off.cir").read_text(encoding="utf-8").splitlines()[:14]
    render("screen_netlist.png", "Get-Content netlists\\transistor\\nmos_off.cir -TotalCount 14", net)

    exe = find_ngspice(CFG.raw.get("ngspice", {}).get("path"))
    r = subprocess.run([str(exe), "-b", str(CFG.netlists / "gates" / "NAND2.cir")], capture_output=True, text=True,
                       timeout=120, cwd=ROOT)
    out = [ln for ln in (r.stdout + r.stderr).splitlines() if ln.strip()]
    render("screen_ngspice.png", "ngspice_con -b netlists\\gates\\NAND2.cir", out)

    df = pd.read_csv(CFG.data / "transistor" / "nmos_off.csv", nrows=5)
    txt = df.to_string(index=False, float_format=lambda v: f"{v:.4g}").splitlines()
    render("screen_dataset.png", "python -c \"import pandas as pd; print(pd.read_csv('data/transistor/nmos_off.csv', nrows=5))\"",
           txt, width_chars=max(len(t) for t in txt) + 2)

    g = pd.read_csv(CFG.results / "final" / "gate_predictions_partB.csv", dtype={"inputs": str})
    cols = ["gate", "inputs", "corner_id", "spice", "i_leak", "spice_node_x", "ml_input_node_x",
            "spice_node_out", "ml_input_node_out"]
    sel = g[(g.gate == "NAND2") & (g.corner_id == 0)][cols].rename(columns={"spice": "I_spice", "i_leak": "I_ml"})
    txt = sel.to_string(index=False, float_format=lambda v: f"{v:.5g}").splitlines()
    render("screen_nodes.png", "python -c \"...read results/final/gate_predictions_partB.csv (NAND2, corner 0)\"",
           txt, width_chars=max(len(t) for t in txt) + 2)

    log = (CFG.results / "c1_cin1_log.txt").read_text(encoding="utf-8").splitlines()
    log = [ln for ln in log if "chunk" not in ln and ln.strip() and "EXIT=" not in ln][:17]
    render("screen_cla_cin1.png", "python scripts\\c1_cla_spice.py --cin 1", log)

    ver = (CFG.results / "run_logs" / "verify_summary.md").read_text(encoding="utf-8").splitlines()
    render("screen_verify.png", "python run_all.py --mode verify", [ln for ln in ver if ln.strip()], width_chars=70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
