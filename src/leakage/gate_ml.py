"""Part B steps 4-5 (and Part C): hierarchical gate leakage prediction from the four
trained TRANSISTOR models. No gate-level model is trained, and no SPICE simulation
is run during inference (only the transistor ML models are evaluated).

Inputs per (gate, input pattern, corner)
  * corner: VDD, T, NMOS and PMOS process parameters      (data/gates/corners.csv)
  * input pins at exactly 0 or VDD                          (from the pattern)
  * internal-node and output voltages (out, x, y, n1)       <- SPICE-recorded (Part B step 3).
    The PDF explicitly allows using the SPICE-recorded node voltages at the gate-validation
    stage; the prediction is therefore NOT independent of SPICE node voltages.

Terminal voltages: each transistor's Vd, Vg, Vs, Vb come from the node it is wired to
(gate topology in leakage.gates): "0" -> 0, "vdd" -> VDD, a/b -> 0 or VDD, internal
nodes -> recorded voltage. Bodies: NMOS -> 0, PMOS -> VDD.

ON/OFF model selection (the spec's cases are defined by the gate bias):
  NMOS: Vg = VDD -> nmos_on, Vg = 0 -> nmos_off;  PMOS: Vg = 0 -> pmos_on, Vg = VDD -> pmos_off.
  Ambiguities, handled explicitly:
  1. Gates driven by an internal node (the inverter in AND2/OR2 is driven by n1) see a
     voltage a few uV from a rail, not exactly 0/VDD. They are classified by the nearest
     rail and the model input Vg is snapped to that rail (the models were trained with Vg
     exactly at a rail). The snap size is recorded (`vg_snap`); a gate voltage farther than
     SNAP_LIMIT * VDD from both rails raises an error.
  2. "ON" means gate-biased ON, not necessarily conducting: e.g. NAND2 "10", MNA has
     Vg = VDD but its source sits at x ~ 0.77 V (Vgs below threshold). The ON model is
     still the right one: its training data has Vs anywhere in [0, VDD].

Leakage accounting (same definition as the SPICE reference, P_leak / VDD):
  P_dev  = (Vd - Vs)*Id + (Vg - Vs)*Ig + (Vb - Vs)*Ib     per transistor (A*V, currents INTO
           the device). For any device obeying KCL this equals sum_t V_t*I_t; using the
           source as reference eliminates the predicted Is, so KCL is enforced per device
           and no terminal current is counted twice.
  P_leak = sum over transistors of P_dev  (Tellegen: = power delivered by the VDD rail and
           the input sources, which is what SPICE reports);  I_leak = P_leak / VDD.
  An ON transistor's channel current enters only through its (tiny) Vds, so the gate
  leakage is dominated by OFF-device sub-threshold, gate-tunnelling and junction currents.
  Diagnostics (not used for I_leak): VDD-rail current = sum of currents into terminals on
  node vdd; input-source currents = sum of currents into terminals on node a / b.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .gates import TERMS, Gate, build_gates
from .inference import LeakagePredictor

NODE_COLUMNS = {"out": "v_out", "x": "v_x", "y": "v_y", "n1": "v_n1"}
PROCESS = ("vth0", "toxe", "ndep", "u0", "xj")
SNAP_LIMIT = 0.05          # x VDD: max distance of a gate voltage from a rail


def device_power(df: pd.DataFrame, prefix: str = "") -> pd.Series:
    """(Vd-Vs)*Id + (Vg-Vs)*Ig + (Vb-Vs)*Ib with currents columns prefix+id/ig/ib.
    Uses the actual node voltages (vg_node), not the snapped model input."""
    return ((df.vd - df.vs) * df[f"{prefix}id"] + (df.vg_node - df.vs) * df[f"{prefix}ig"]
            + (df.vb - df.vs) * df[f"{prefix}ib"])


def terminal_table(records: pd.DataFrame, gates: dict[str, Gate] | None = None, cfg=None) -> pd.DataFrame:
    """One row per transistor instance with its terminal voltages, PVT and model case.

    records: rows with gate, inputs, corner_id, vdd, temp, n_*/p_* process params and the
    recorded node voltages v_out (+ v_x / v_y / v_n1 where the gate has them)."""
    if gates is None:
        from .config import load_config
        gates = build_gates(cfg or load_config())
    rows = []
    for r in records.itertuples(index=False):
        rec = r._asdict()
        g = gates[rec["gate"]]
        vdd = float(rec["vdd"])
        bits = str(rec["inputs"]).zfill(len(g.inputs))
        volt = {"0": 0.0, "vdd": vdd}
        volt.update({pin: (vdd if b == "1" else 0.0) for pin, b in zip(g.inputs, bits)})
        for n in g.nodes:
            v = rec.get(NODE_COLUMNS[n])
            if v is None or not np.isfinite(v):
                raise ValueError(f"{g.name} {bits} corner {rec['corner_id']}: missing voltage of node {n}")
            volt[n] = float(v)
        for d in g.devices:
            vg_node = volt[d.g]
            high = vg_node >= vdd / 2
            vg = vdd if high else 0.0
            snap = abs(vg_node - vg)
            if snap > SNAP_LIMIT * vdd:
                raise ValueError(f"{g.name} {bits} corner {rec['corner_id']} {d.name}: gate voltage "
                                 f"{vg_node:.4g} V is not near a rail; ON/OFF model undefined")
            on = high if d.dev == "nmos" else not high
            pre = "n_" if d.dev == "nmos" else "p_"
            rows.append({"gate": g.name, "inputs": bits, "corner_id": int(rec["corner_id"]),
                         "device": d.name, "type": d.dev, "state": "on" if on else "off",
                         "case": f"{d.dev}_{'on' if on else 'off'}", "w": d.w,
                         **{f"node_{t}": d.node(t) for t in TERMS},
                         "vd": volt[d.d], "vg_node": vg_node, "vg": vg, "vg_snap": snap,
                         "vs": volt[d.s], "vb": volt[d.b], "vdd": vdd, "temp": float(rec["temp"]),
                         **{p: float(rec[pre + p]) for p in PROCESS}})
    return pd.DataFrame(rows)


def predict_devices(table: pd.DataFrame, predictor: LeakagePredictor) -> pd.DataFrame:
    """Add predicted id, ig, is, ib (A, into the device) and the extrapolation flag."""
    out = table.copy()
    for col in ("id", "ig", "is", "ib"):
        out[col] = np.nan
    out["extrapolated"] = False
    for (dev, state), grp in out.groupby(["type", "state"]):
        pred = predictor.predict_frame(dev, grp[["vdd", "temp", "w", "vd", "vs", "vg", "vb", *PROCESS]],
                                       state=state)
        out.loc[grp.index, ["id", "ig", "is", "ib"]] = pred[["id", "ig", "is", "ib"]].to_numpy()
        out.loc[grp.index, "extrapolated"] = pred["extrapolated"].to_numpy()
    return out


def gate_leakage(dev: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """Aggregate per (gate, inputs, corner): P_leak, I_leak = P_leak/VDD, and diagnostics."""
    d = dev.copy()
    d["p_dev"] = device_power(d, prefix)
    # currents into terminals on the vdd / input nodes (diagnostic supply accounting)
    for node in ("vdd", "a", "b"):
        d[f"i_node_{node}"] = sum(np.where(d[f"node_{t}"] == node, d[f"{prefix}i{t}"], 0.0) for t in TERMS)
    agg = d.groupby(["gate", "inputs", "corner_id"], sort=False).agg(
        vdd=("vdd", "first"), p_leak=("p_dev", "sum"),
        i_vdd=("i_node_vdd", "sum"), i_in_a=("i_node_a", "sum"), i_in_b=("i_node_b", "sum"),
        n_devices=("device", "size")).reset_index()
    agg["i_leak"] = agg.p_leak / agg.vdd
    return agg


def predict_gate_records(records: pd.DataFrame, predictor: LeakagePredictor | None = None,
                         gates: dict[str, Gate] | None = None):
    """Full pipeline: records (corner + recorded node voltages) -> (gate_df, device_df)."""
    predictor = predictor or LeakagePredictor()
    table = terminal_table(records, gates, predictor.cfg)
    dev = predict_devices(table, predictor)
    gate = gate_leakage(dev)
    ext = dev.groupby(["gate", "inputs", "corner_id"], sort=False).extrapolated.any().reset_index()
    snap = dev.groupby(["gate", "inputs", "corner_id"], sort=False).vg_snap.max().reset_index()
    gate = gate.merge(ext, on=["gate", "inputs", "corner_id"]).merge(snap, on=["gate", "inputs", "corner_id"])
    return gate, dev


class GateLeakageModel:
    """Reusable gate model for Part C: leakage of (gate, input bits, corner_id) from the
    transistor ML models, using the Part B SPICE-recorded node voltages of that exact
    gate / input / corner (the PDF's gate model). Results are cached."""

    def __init__(self, records: pd.DataFrame | None = None, predictor: LeakagePredictor | None = None):
        self.predictor = predictor or LeakagePredictor()
        cfg = self.predictor.cfg
        if records is None:
            records = pd.read_csv(cfg.data / "gates" / "gate_spice_results.csv", dtype={"inputs": str})
        self.records = records.set_index(["gate", "inputs", "corner_id"]).sort_index()
        self.gates = build_gates(cfg)
        self._cache: dict = {}

    def leakage(self, gate: str, bits: str, corner_ids) -> pd.Series:
        """Predicted I_leak (A) for one gate type / input pattern at the given corners."""
        corner_ids = list(np.atleast_1d(corner_ids))
        missing = [c for c in corner_ids if (gate, bits, c) not in self._cache]
        if missing:
            rec = self.records.loc[[(gate, bits, c) for c in missing]].reset_index()
            g, _ = predict_gate_records(rec, self.predictor, self.gates)
            for row in g.itertuples(index=False):
                self._cache[(row.gate, row.inputs, row.corner_id)] = row.i_leak
        return pd.Series([self._cache[(gate, bits, c)] for c in corner_ids], index=corner_ids, name="i_leak")
