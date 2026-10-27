import numpy as np
import pandas as pd
import pytest

from leakage.config import load_config
from leakage.dataset import (TooManyFailures, assemble, generate_samples, ids_needed,
                             input_columns, load_shards, shard_dir, simulate_ids, validate)
from leakage.simulator import NgspiceSimulator
from leakage.spice import find_ngspice

CFG = load_config()
needs_ngspice = pytest.mark.skipif(find_ngspice(CFG.ngspice_path) is None,
                                   reason="ngspice not found")


def fake_shards(case, ids, failed=()):
    df = generate_samples(CFG, case, ids)
    for i, c in enumerate(["id", "ig", "is", "ib"]):
        df[c] = [1e-9, -1e-12, -0.999e-9 + 1e-12, 0.0][i] or 1e-15
    df["is"] = -(df["id"] + df["ig"] + df["ib"])
    df["status"] = np.where(df.sample_id.isin(failed), "failed", "ok")
    df.loc[df.status == "failed", ["id", "ig", "is", "ib"]] = np.nan
    df["messages"] = ""
    return df[["sample_id", *CFG.columns, "status", "messages"]]


# ------------------------------------------------------------------ sampling
def test_sample_ids_are_stable_prefixes():
    a = generate_samples(CFG, "nmos_on", range(20))
    b = generate_samples(CFG, "nmos_on", range(2500))
    pd.testing.assert_frame_equal(a, b.head(20))
    c = generate_samples(CFG, "nmos_on", [1005, 2400])
    pd.testing.assert_frame_equal(c, b[b.sample_id.isin([1005, 2400])].reset_index(drop=True))


def test_sampling_deterministic_and_case_specific():
    a = generate_samples(CFG, "pmos_off", range(50))
    pd.testing.assert_frame_equal(a, generate_samples(CFG, "pmos_off", range(50)))
    assert not a[input_columns(CFG)].equals(generate_samples(CFG, "pmos_on", range(50))[input_columns(CFG)])
    assert list(a.columns) == ["sample_id", *input_columns(CFG)]


@pytest.mark.parametrize("case", list(CFG.cases))
def test_generated_inputs_pass_range_checks(case):
    df = fake_shards(case, range(3000))
    checks = validate(CFG, case, df[list(CFG.columns)], 3000, df.sample_id)
    assert all(ok for ok, _ in checks), [t for ok, t in checks if not ok]


# ------------------------------------------------------------------ resume / top-up
def test_ids_needed_progression():
    empty = fake_shards("nmos_off", [])
    assert list(ids_needed(empty, 10, 50)) == list(range(10))
    part = fake_shards("nmos_off", [0, 1, 2, 5])
    assert list(ids_needed(part, 10, 50)) == [3, 4, 6, 7, 8, 9]
    full_with_fail = fake_shards("nmos_off", range(10), failed=[3, 7])
    assert list(ids_needed(full_with_fail, 10, 50)) == [10, 11]
    topped = fake_shards("nmos_off", range(12), failed=[3, 7])
    assert len(ids_needed(topped, 10, 50)) == 0
    with pytest.raises(TooManyFailures):
        ids_needed(fake_shards("nmos_off", range(10), failed=range(10)), 10, 5)


def test_assemble_takes_first_n_valid_and_logs_failures():
    sh = fake_shards("nmos_off", range(12), failed=[3, 7])
    final, index, failures = assemble(CFG, sh, 10)
    assert list(final.columns) == list(CFG.columns)
    assert list(index.sample_id) == [0, 1, 2, 4, 5, 6, 8, 9, 10, 11]
    assert list(failures.sample_id) == [3, 7]
    assert np.isfinite(final[["id", "ig", "is", "ib"]].to_numpy()).all()


# ------------------------------------------------------------------ validation catches problems
def test_validate_catches_bad_data():
    df = fake_shards("nmos_on", range(100))[list(CFG.columns)]
    bad = df.copy(); bad.loc[0, "id"] = np.nan
    assert not all(ok for ok, _ in validate(CFG, "nmos_on", bad, 100))
    bad = df.copy(); bad.loc[0, "vdd"] = 1.5
    assert not all(ok for ok, _ in validate(CFG, "nmos_on", bad, 100))
    bad = df.copy(); bad.loc[0, "vb"] = 0.5
    assert not all(ok for ok, _ in validate(CFG, "nmos_on", bad, 100))
    bad = pd.concat([df, df.head(1)], ignore_index=True)
    assert not all(ok for ok, _ in validate(CFG, "nmos_on", bad, 101))
    assert not all(ok for ok, _ in validate(CFG, "nmos_on", df[list(reversed(CFG.columns))], 100))
    assert not all(ok for ok, _ in validate(CFG, "nmos_on", df, 99))


# ------------------------------------------------------------------ end-to-end with ngspice
@needs_ngspice
def test_simulate_resume_without_duplicates(tmp_path):
    sim = NgspiceSimulator.from_config(CFG, workers=4, chunk=5)
    case = "nmos_off"
    simulate_ids(CFG, case, ids_needed(load_shards(CFG, tmp_path, case), 30, 50),
                 tmp_path, sim, batch_size=10)
    sh = load_shards(CFG, tmp_path, case)
    assert list(sh.sample_id) == list(range(30)) and (sh.status == "ok").all()
    assert len(ids_needed(sh, 30, 50)) == 0                   # nothing left: resume is a no-op

    # simulate an interruption: lose one shard, only its ids are redone
    victim = sorted(shard_dir(tmp_path, case).glob("ids_*.csv"))[1]
    victim.unlink()
    todo = ids_needed(load_shards(CFG, tmp_path, case), 30, 50)
    assert list(todo) == list(range(10, 20))
    simulate_ids(CFG, case, todo, tmp_path, sim, batch_size=10)
    sh2 = load_shards(CFG, tmp_path, case)
    assert sh2.sample_id.is_unique and len(sh2) == 30
    pd.testing.assert_frame_equal(sh.reset_index(drop=True), sh2.reset_index(drop=True))

    final, index, failures = assemble(CFG, sh2, 30)
    assert all(ok for ok, _ in validate(CFG, case, final, 30, index.sample_id))
    assert len(failures) == 0
