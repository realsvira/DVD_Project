"""ngspice + model-card helpers.

Conventions
-----------
* Terminal currents are POSITIVE when flowing INTO the device terminal.
  ngspice reports i(Vx) from the source's + node through the source to its
  - node. With each terminal source wired terminal(+) -> ground(-), the
  current into the device terminal is therefore -i(Vx).
* Process variation is applied by writing the varied parameters into the
  model card text of a fresh netlist. Do NOT use `altermod` for toxe:
  ngspice 44.2 (BSIM4) accepts the new value but does not recompute the
  oxide-derived quantities, giving gate/channel currents off by up to ~5x
  (verified in this project). vth0/ndep/u0/xj via altermod were fine, but a
  fresh model card is used for everything to keep one trusted code path.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

NUM_RE = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_MODEL_SPLIT = re.compile(r"(?im)(^[ \t]*\.model[ \t]+)")


# ===================================================================== model cards
def _model_blocks(text: str) -> list[str]:
    """['header', '.model ', block1, '.model ', block2, ...]"""
    return _MODEL_SPLIT.split(text)


def model_info(model_file: Path) -> dict[str, dict]:
    """{'nmos': {'type': 'nmos', 'level': 54, 'params': {name: str value}}, ...}"""
    parts = _model_blocks(Path(model_file).read_text())
    info = {}
    for i in range(2, len(parts), 2):
        blk = parts[i]
        tok = blk.split()
        name, mtype = tok[0].lower(), tok[1].lower()
        params = {k.lower(): v for k, v in
                  re.findall(rf"(?<![A-Za-z0-9_])([A-Za-z][A-Za-z0-9_]*)\s*=\s*({NUM_RE})", blk)}
        info[name] = {"type": mtype, "level": int(float(params.get("level", "nan"))),
                      "params": params}
    return info


def read_model_nominals(model_file: Path, params) -> dict[str, dict[str, float]]:
    """Nominal values of the selected params for every model in the file."""
    out = {}
    for name, d in model_info(model_file).items():
        missing = [p for p in params if p.lower() not in d["params"]]
        if missing:
            raise ValueError(f"{name}: parameters {missing} not found in {model_file}")
        out[name] = {p: float(d["params"][p.lower()]) for p in params}
    return out


def model_text_with(model_file: Path, overrides: dict[str, dict[str, float]] | None = None,
                    only: str | None = None) -> str:
    """Model-card text with parameter overrides applied.

    overrides: {'nmos': {'toxe': 1.3e-9, ...}, 'pmos': {...}}
    only:      return just this model's card (e.g. 'nmos').
    """
    overrides = {k.lower(): v for k, v in (overrides or {}).items()}
    parts = _model_blocks(Path(model_file).read_text())
    out = [] if only else [parts[0]]
    for i in range(2, len(parts), 2):
        blk = parts[i]
        name = blk.split()[0].lower()
        for p, v in overrides.get(name, {}).items():
            blk, n = re.subn(rf"(?i)(?<![A-Za-z0-9_])({re.escape(p)}\s*=\s*){NUM_RE}",
                             lambda m: f"{m.group(1)}{v:.10g}", blk, count=1)
            if n != 1:
                raise ValueError(f"could not set {p} on model {name}")
        if only is None or name == only.lower():
            out += [parts[i - 1].lstrip("\r\n"), blk]
    if only and len(out) == 0:
        raise ValueError(f"model {only} not in {model_file}")
    return "".join(out).rstrip() + "\n"


def spice_path(p: Path | str) -> str:
    """Path string safe for ngspice netlists on Windows (forward slashes, quoted)."""
    return '"' + str(Path(p).resolve()).replace("\\", "/") + '"'


# ===================================================================== locating ngspice
_COMMON_DIRS = [
    r"C:\Spice64\bin",
    r"C:\Program Files\Spice64\bin",
    r"C:\Program Files\ngspice\bin",
    r"C:\Program Files (x86)\Spice64\bin",
    r"D:\VLSI\NGSPICE\ngspice-44.2_64\Spice64\bin",
]


def find_ngspice(configured: Path | str | None = None) -> Path | None:
    """Return the console ngspice executable, or None.

    Order: NGSPICE env var -> configured path -> PATH -> common folders.
    Prefers ngspice_con.exe (console) over ngspice.exe (GUI) on Windows.
    """
    candidates: list[Path] = []
    if os.environ.get("NGSPICE"):
        candidates.append(Path(os.environ["NGSPICE"]))
    if configured:
        candidates.append(Path(configured))
    for exe in ("ngspice_con", "ngspice"):
        w = shutil.which(exe)
        if w:
            candidates.append(Path(w))
    for d in _COMMON_DIRS:
        candidates += [Path(d) / "ngspice_con.exe", Path(d) / "ngspice.exe"]
    for c in candidates:
        if c.is_file():
            # prefer the console build sitting next to a GUI build
            con = c.with_name("ngspice_con.exe")
            return con if c.name.lower() == "ngspice.exe" and con.is_file() else c
    return None


def ngspice_version(exe: Path) -> str:
    r = subprocess.run([str(exe), "-v"], capture_output=True, text=True, timeout=60)
    m = re.search(r"ngspice-?\s*([\w.+-]+)", r.stdout + r.stderr, re.I)
    return m.group(1) if m else "unknown"


# ===================================================================== running
def run_ngspice(exe: Path, netlist: str, extra_files: dict[str, str] | None = None,
                timeout: int = 3600) -> str:
    """Run `ngspice -b` on netlist text in a temporary folder; return stdout+stderr.

    extra_files: {filename: text} written beside the netlist (e.g. circuits
    that a .control driver `source`s).
    """
    with tempfile.TemporaryDirectory(prefix="ngspice_") as td:
        tdp = Path(td)
        for name, txt in (extra_files or {}).items():
            (tdp / name).write_text(txt)
        f = tdp / "run.cir"
        f.write_text(netlist)
        r = subprocess.run([str(exe), "-b", str(f)], capture_output=True, text=True,
                           timeout=timeout, cwd=td)
    out = r.stdout + "\n" + r.stderr
    if r.returncode != 0:
        raise RuntimeError(f"ngspice exited with {r.returncode}:\n{out[-3000:]}")
    return out


def parse_raw_ascii(text: str) -> list[dict]:
    """Parse an ngspice ASCII raw file (`set filetype=ascii` + `write`).

    Returns one dict per plot: {'title', 'plotname', 'variables': [names],
    'values': np.ndarray (n_points, n_vars)}. Variable names are lower-case.
    Real-valued plots only (op / dc / tran).
    """
    import numpy as np
    plots, lines, i = [], text.splitlines(), 0
    while i < len(lines):
        if not lines[i].startswith("Title:"):
            i += 1
            continue
        hdr = {"title": lines[i][6:].strip()}
        i += 1
        while i < len(lines) and not lines[i].startswith("Variables:"):
            key, _, val = lines[i].partition(":")
            hdr[key.strip().lower()] = val.strip()
            i += 1
        if "complex" in hdr.get("flags", ""):
            raise ValueError("complex raw files are not supported")
        nvar, npts = int(hdr["no. variables"]), int(hdr["no. points"])
        names = [lines[i + 1 + j].split()[1].lower() for j in range(nvar)]
        i += 1 + nvar
        if not lines[i].startswith("Values:"):
            raise ValueError("malformed raw file: 'Values:' not found")
        i += 1
        tokens: list[str] = []
        while len(tokens) < npts * (nvar + 1) and i < len(lines):
            tokens += lines[i].split()
            i += 1
        if len(tokens) < npts * (nvar + 1):
            raise ValueError("raw file truncated")
        arr = np.array(tokens[: npts * (nvar + 1)], dtype=float).reshape(npts, nvar + 1)[:, 1:]
        plots.append({"title": hdr["title"], "plotname": hdr.get("plotname", ""),
                      "variables": names, "values": arr})
    return plots


_PRINT_RE = re.compile(rf"^\s*([A-Za-z]+\([^)]*\))\s*=\s*({NUM_RE})\s*$")


def parse_prints(out: str, tag: str = "@@S") -> dict:
    """Parse `print` output.

    If the output contains lines `<tag> <k>` (from `echo <tag> <k>`), values are
    grouped per k: {k: {'i(vd)': ..}}. Otherwise returns {None: {...}}.
    """
    res: dict = {}
    cur = None
    res[cur] = {}
    for line in out.splitlines():
        s = line.strip()
        if s.startswith(tag + " "):
            cur = int(s.split()[1])
            res[cur] = {}
            continue
        m = _PRINT_RE.match(s)
        if m:
            res[cur][m.group(1).lower()] = float(m.group(2))
    if len(res) > 1 and not res[None]:
        del res[None]
    return res


def simulator_errors(out: str) -> list[str]:
    """Lines that indicate a real simulator problem."""
    bad = ("error", "singular matrix", "timestep too small", "no convergence",
           "unknown model", "could not find", "fatal")
    return [l.strip() for l in out.splitlines() if any(b in l.lower() for b in bad)]
