"""Load config/project.toml into a typed, path-resolved object.

Usage:
    from leakage.config import load_config
    cfg = load_config()
    cfg.spice_model_file   # absolute pathlib.Path
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "project.toml"


@dataclass(frozen=True)
class Case:
    name: str
    device: str      # "nmos" | "pmos"
    vg_frac: float   # gate voltage  = vg_frac * VDD
    vb_frac: float   # body voltage  = vb_frac * VDD


@dataclass(frozen=True)
class Config:
    raw: dict
    root: Path
    # assignment
    model_name: str
    vdd_nominal: float
    lmin: float
    test_circuit: str
    # paths
    spice_model_file: Path
    spice_model_sha256: str
    netlists: Path
    data: Path
    trained_models: Path
    results: Path
    figures: Path
    reports: Path
    inputs_extracted: Path
    # ngspice
    ngspice_path: Path
    spice_options: str
    ngspice_timeout_s: int
    ngspice_workers: int
    ngspice_chunk: int
    # sampling
    n_samples: int
    vdd_tolerance: float
    w_range: tuple[float, float]
    temp_range: tuple[float, float]
    process_params: tuple[str, ...]
    process_tolerance: float
    cases: dict[str, Case] = field(default_factory=dict)
    columns: tuple[str, ...] = ()
    split: dict[str, float] = field(default_factory=dict)
    seeds: dict[str, int] = field(default_factory=dict)

    @property
    def vdd_range(self) -> tuple[float, float]:
        return (self.vdd_nominal * (1 - self.vdd_tolerance),
                self.vdd_nominal * (1 + self.vdd_tolerance))


def _resolve(root: Path, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (root / path).resolve()


def load_config(path: Path | str = DEFAULT_CONFIG) -> Config:
    path = Path(path)
    with path.open("rb") as f:
        raw = tomllib.load(f)
    root = PROJECT_ROOT
    a, p, n, s = raw["assignment"], raw["paths"], raw["ngspice"], raw["sampling"]
    lmin = float(a["lmin"])
    ngspice = os.environ.get("NGSPICE") or n["path"]

    cfg = Config(
        raw=raw, root=root,
        model_name=a["model_name"], vdd_nominal=float(a["vdd_nominal"]),
        lmin=lmin, test_circuit=a["test_circuit"],
        spice_model_file=_resolve(root, p["spice_model_file"]),
        spice_model_sha256=p.get("spice_model_sha256", ""),
        netlists=_resolve(root, p["netlists"]), data=_resolve(root, p["data"]),
        trained_models=_resolve(root, p["trained_models"]),
        results=_resolve(root, p["results"]), figures=_resolve(root, p["figures"]),
        reports=_resolve(root, p["reports"]),
        inputs_extracted=_resolve(root, p["inputs_extracted"]),
        ngspice_path=Path(ngspice), spice_options=n["options"],
        ngspice_timeout_s=int(n.get("timeout_s", 3600)),
        ngspice_workers=int(n.get("workers", 0)) or max(1, (os.cpu_count() or 2) - 1),
        ngspice_chunk=int(n.get("chunk", 200)),
        n_samples=int(s["n_samples"]), vdd_tolerance=float(s["vdd_tolerance"]),
        w_range=(s["w_min_factor"] * lmin, s["w_max_factor"] * lmin),
        temp_range=(float(s["temp_min_c"]), float(s["temp_max_c"])),
        process_params=tuple(s["process_params"]),
        process_tolerance=float(s["process_tolerance"]),
        cases={k: Case(k, v["device"], float(v["vg_frac"]), float(v["vb_frac"]))
               for k, v in raw["cases"].items()},
        columns=tuple(raw["dataset"]["columns"]),
        split=dict(raw["split"]),
        seeds={k: int(v) for k, v in raw["seeds"].items()},
    )
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    if abs(sum(cfg.split.values()) - 1.0) > 1e-9:
        raise ValueError(f"split fractions must sum to 1, got {cfg.split}")
    for c in cfg.cases.values():
        if c.device not in ("nmos", "pmos"):
            raise ValueError(f"case {c.name}: unknown device {c.device}")
    if cfg.model_name not in cfg.spice_model_file.stem:
        raise ValueError(f"model_name {cfg.model_name} does not match {cfg.spice_model_file.name}")
