# Part A1 dataset generation summary

- Created: 2026-11-13T18:45:46  |  command: `a1_generate_datasets.py --small --skip-smoke --no-progress`
- Model `models\45nm_HP.pm` (sha256 9ab66a7bc4a5...), ngspice 44.2
- Git commit 2366e6ebbc (dirty)
- Output folder: `data\transistor_small`
- This run: 0 simulations in 0.1 s

| Case | Rows | Simulated | Succeeded | Failed | Sample ids | max abs KCL sum (A) | median abs Id (A) | Checks |
|---|---:|---:|---:|---:|---|---:|---:|---|
| nmos_on | 200 | 200 | 200 | 0 | 0..199 | 8.50e-16 | 3.47e-05 | all pass |
| nmos_off | 200 | 200 | 200 | 0 | 0..199 | 6.57e-24 | 8.76e-12 | all pass |
| pmos_on | 200 | 200 | 200 | 0 | 0..199 | 1.23e-16 | 2.63e-05 | all pass |
| pmos_off | 200 | 200 | 200 | 0 | 0..199 | 8.09e-16 | 1.92e-11 | all pass |

Simulator warnings (from successful batches; counts per message):

- nmos_on: Warning: toxe, toxp and dtox all given and toxe != toxp + dtox; dtox ignored. ×200
- nmos_off: Warning: toxe, toxp and dtox all given and toxe != toxp + dtox; dtox ignored. ×200
- pmos_on: Warning: toxe, toxp and dtox all given and toxe != toxp + dtox; dtox ignored. ×200
- pmos_off: Warning: toxe, toxp and dtox all given and toxe != toxp + dtox; dtox ignored. ×200

**Totals:** 800 simulated, 800 succeeded, 0 failed (see `failures.csv`).
**Validation:** all checks passed.
