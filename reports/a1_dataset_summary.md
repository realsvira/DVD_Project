# Part A1 dataset generation summary

- Created: 2026-11-12T23:40:04  |  command: `a1_generate_datasets.py --no-progress`
- Model `models\45nm_HP.pm` (sha256 9ab66a7bc4a5...), ngspice 44.2
- Git commit 9658349429 (dirty)
- Output folder: `data\transistor`
- This run: 40000 simulations in 4.7 min (143 sim/s)

| Case | Rows | Simulated | Succeeded | Failed | Sample ids | max abs KCL sum (A) | median abs Id (A) | Checks |
|---|---:|---:|---:|---:|---|---:|---:|---|
| nmos_on | 10000 | 10000 | 10000 | 0 | 0..9999 | 1.90e-15 | 3.59e-05 | all pass |
| nmos_off | 10000 | 10000 | 10000 | 0 | 0..9999 | 1.87e-23 | 7.77e-12 | all pass |
| pmos_on | 10000 | 10000 | 10000 | 0 | 0..9999 | 1.86e-16 | 1.68e-05 | all pass |
| pmos_off | 10000 | 10000 | 10000 | 0 | 0..9999 | 1.51e-15 | 1.86e-11 | all pass |

Simulator warnings (from successful batches; counts per message):

- nmos_on: Warning: toxe, toxp and dtox all given and toxe != toxp + dtox; dtox ignored. ×10000
- nmos_off: Warning: toxe, toxp and dtox all given and toxe != toxp + dtox; dtox ignored. ×10000
- pmos_on: Warning: toxe, toxp and dtox all given and toxe != toxp + dtox; dtox ignored. ×10000
- pmos_off: Warning: toxe, toxp and dtox all given and toxe != toxp + dtox; dtox ignored. ×10000

**Totals:** 40000 simulated, 40000 succeeded, 0 failed (see `failures.csv`).
**Validation:** all checks passed.
