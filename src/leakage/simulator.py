"""Reusable batch DC operating-point simulator on top of ngspice.

Used by every stage: single transistors (Part A), gates (Part B) and the
test circuit (Part C). Callers supply complete circuit texts (no .control
section, ending in .end) and the vectors to record, e.g. ["i(vd)", "v(out)"].

How it works
------------
* Circuits are split into chunks; each chunk runs in ONE ngspice process
  (`ngspice -b -o run.log driver.cir`) and chunks run in parallel threads.
* The driver loops: `source s<k>.cir` -> `op` -> `write o<k>.raw <vectors>`
  -> `remcirc` -> `destroy all`. Values are read from the ASCII raw files
  (structured, full double precision), never from the printed log.
* Success is decided per circuit from its own raw file: a circuit that fails
  (parse error, no convergence, missing vector, non-finite value) writes no
  usable raw file, is marked failed, and the other circuits are unaffected.
* Diagnostics: ngspice writes errors to stderr (unbuffered) and `echo`
  markers to stdout (buffered), so a shared log cannot be split per circuit
  reliably. Therefore every failed circuit is re-run ALONE in its own process;
  that log belongs to it only and becomes its `messages`. (If it succeeds when
  run alone, that result is used and the fact is noted.) Warnings seen in a
  batch log are kept, unattributed, in `OpResult.batch_warnings`.
* If a whole ngspice process dies or times out, its circuits are re-run one
  per process, so one bad circuit cannot take down its neighbours.
"""
from __future__ import annotations

import math
import subprocess
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from .spice import find_ngspice, parse_raw_ascii

END_MARK = "@@END_OF_BATCH"
_BENIGN = ("no compatibility mode selected", "simulation executed from .control",
           "comments and warnings go to log-file")
_ISSUE_WORDS = ("warning", "error", "singular", "timestep too small", "convergence",
                "failed", "interrupted", "not parsed", "could not", "can't find", "cannot")


@dataclass
class OpResult:
    """values:   DataFrame, one row per circuit, one column per vector (NaN if missing).
    status:   'ok' | 'failed' per circuit.
    messages: per-circuit diagnostics (exact for failed circuits).
    batch_warnings: Counter of warning lines from batch logs (not attributed)."""
    values: pd.DataFrame
    status: pd.Series
    messages: list[list[str]] = field(default_factory=list)
    batch_warnings: Counter = field(default_factory=Counter)
    seconds: float = 0.0

    @property
    def n_failed(self) -> int:
        return int((self.status != "ok").sum())

    def failures(self) -> pd.DataFrame:
        idx = np.flatnonzero(self.status.to_numpy() != "ok")
        return pd.DataFrame({"index": idx,
                             "messages": [" | ".join(self.messages[i]) for i in idx]})


@dataclass
class _Chunk:
    vals: np.ndarray
    ok: np.ndarray
    msgs: list[list[str]]
    warnings: Counter


class NgspiceSimulator:
    def __init__(self, exe: Path | str | None = None, workers: int = 4, chunk: int = 200,
                 timeout_s: int = 3600):
        self.exe = Path(exe) if exe else find_ngspice()
        if self.exe is None or not self.exe.is_file():
            raise FileNotFoundError("ngspice not found; set NGSPICE or [ngspice].path")
        self.workers, self.chunk, self.timeout_s = max(1, workers), max(1, chunk), timeout_s

    @classmethod
    def from_config(cls, cfg, **overrides) -> "NgspiceSimulator":
        kw = dict(exe=find_ngspice(cfg.ngspice_path), workers=cfg.ngspice_workers,
                  chunk=cfg.ngspice_chunk, timeout_s=cfg.ngspice_timeout_s)
        kw.update(overrides)
        return cls(**kw)

    # ------------------------------------------------------------------ public
    def run_op(self, circuits: Sequence[str], vectors: Sequence[str],
               progress: Callable[[int, int, float], None] | None = None) -> OpResult:
        """DC operating point of every circuit; record `vectors` for each."""
        vectors = [v.lower() for v in vectors]
        n = len(circuits)
        vals = np.full((n, len(vectors)), np.nan)
        ok = np.zeros(n, dtype=bool)
        msgs: list[list[str]] = [[] for _ in range(n)]
        warn: Counter = Counter()
        t0 = time.perf_counter()
        starts = list(range(0, n, self.chunk))

        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            futs = {ex.submit(self._run_chunk, circuits[s:s + self.chunk], vectors): s
                    for s in starts}
            for done, f in enumerate(as_completed(futs), 1):
                s, c = futs[f], f.result()
                m = len(c.ok)
                vals[s:s + m], ok[s:s + m], msgs[s:s + m] = c.vals, c.ok, c.msgs
                warn.update(c.warnings)
                if progress:
                    progress(done, len(starts), time.perf_counter() - t0)

        return OpResult(values=pd.DataFrame(vals, columns=vectors),
                        status=pd.Series(np.where(ok, "ok", "failed")),
                        messages=msgs, batch_warnings=warn,
                        seconds=time.perf_counter() - t0)

    # ------------------------------------------------------------------ internals
    @staticmethod
    def _driver(n: int, vectors: Sequence[str]) -> str:
        vec = " ".join(vectors)
        lines = ["* batch operating-point driver", ".control", "set filetype=ascii"]
        for k in range(n):
            lines += [f"source s{k}.cir", "op", f"write o{k}.raw {vec}", "remcirc",
                      "destroy all"]
        return "\n".join(lines + [f"echo {END_MARK}", ".endc", ".end", ""])

    def _execute(self, circuits: Sequence[str], vectors: Sequence[str]):
        """One ngspice process. Returns (vals, ok, raw_msgs, log, crashed)."""
        n = len(circuits)
        vals = np.full((n, len(vectors)), np.nan)
        ok = np.zeros(n, dtype=bool)
        raw_msgs: list[list[str]] = [[] for _ in range(n)]
        with tempfile.TemporaryDirectory(prefix="ngspice_") as td:
            tdp = Path(td)
            for k, c in enumerate(circuits):
                (tdp / f"s{k}.cir").write_text(c)
            (tdp / "driver.cir").write_text(self._driver(n, vectors))
            try:
                r = subprocess.run([str(self.exe), "-b", "-o", "run.log", "driver.cir"],
                                   cwd=td, capture_output=True, text=True,
                                   timeout=self.timeout_s)
                logf = tdp / "run.log"
                log = (logf.read_text(errors="replace") if logf.is_file() else "") \
                    + "\n" + r.stdout + "\n" + r.stderr
                crashed = END_MARK not in log
            except subprocess.TimeoutExpired:
                log, crashed = f"ngspice timed out after {self.timeout_s} s", True

            for k in range(n):
                raw = tdp / f"o{k}.raw"
                if not raw.is_file():
                    raw_msgs[k].append("no output written (simulation failed)")
                    continue
                try:
                    plot = parse_raw_ascii(raw.read_text())[0]
                except Exception as e:  # noqa: BLE001
                    raw_msgs[k].append(f"unreadable raw file: {e}")
                    continue
                names, row = plot["variables"], plot["values"][0]
                missing = [v for v in vectors if v not in names]
                for j, v in enumerate(vectors):
                    if v in names:
                        vals[k, j] = row[names.index(v)]
                if missing:
                    raw_msgs[k].append(f"missing vectors {missing}")
                elif not np.isfinite(vals[k]).all():
                    raw_msgs[k].append("non-finite value")
                else:
                    ok[k] = True
        return vals, ok, raw_msgs, log, crashed

    def _run_chunk(self, circuits: Sequence[str], vectors: Sequence[str]) -> _Chunk:
        n = len(circuits)
        vals, ok, raw_msgs, log, crashed = self._execute(circuits, vectors)

        if n == 1:
            msgs = [_issues(log) + raw_msgs[0]] if not ok[0] else [[]]
            if crashed and not ok[0]:
                msgs[0].insert(0, "ngspice process crashed or timed out")
            warn = Counter(_issues(log)) if ok[0] else Counter()
            return _Chunk(vals, ok, msgs, warn)

        msgs: list[list[str]] = [[] for _ in range(n)]
        warn = Counter() if crashed else Counter(_issues(log))
        # Re-run every failed circuit (or all of them after a crash) on its own:
        # exact diagnostics, and protection against one circuit killing a batch.
        redo = range(n) if crashed else np.flatnonzero(~ok)
        for k in redo:
            single = self._run_chunk([circuits[k]], vectors)
            vals[k], ok[k], msgs[k] = single.vals[0], single.ok[0], single.msgs[0]
            if ok[k] and not crashed:
                msgs[k] = ["failed in batch, succeeded when simulated alone"]
            warn.update(single.warnings)
        return _Chunk(vals, ok, msgs, warn)


def _issues(log: str) -> list[str]:
    res = []
    for line in log.splitlines():
        low = line.strip().lower()
        if low and any(w in low for w in _ISSUE_WORDS) and not any(b in low for b in _BENIGN):
            res.append(line.strip())
    return res
