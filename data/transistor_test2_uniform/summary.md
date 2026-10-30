# Part A1 dataset generation summary

- Created: 2026-11-13T08:39:42  |  command: `a1_generate_datasets.py --skip-smoke --no-progress --profile uniform --seed-salt 2 --n 1500 --out data/transistor_test2_uniform`
- Model `models\45nm_HP.pm` (sha256 9ab66a7bc4a5...), ngspice 44.2
- Git commit e5fee494f7 (dirty)
- Output folder: `data\transistor_test2_uniform`
- This run: 6000 simulations in 56.2 s (107 sim/s)

| Case | Rows | Simulated | Succeeded | Failed | Sample ids | max abs KCL sum (A) | median abs Id (A) | Checks |
|---|---:|---:|---:|---:|---|---:|---:|---|
| nmos_on | 1500 | 1500 | 1500 | 0 | 0..1499 | 1.05e-15 | 3.99e-05 | all pass |
| nmos_off | 1500 | 1500 | 1500 | 0 | 0..1499 | 2.44e-24 | 7.66e-12 | all pass |
| pmos_on | 1500 | 1500 | 1500 | 0 | 0..1499 | 1.62e-16 | 2.02e-05 | all pass |
| pmos_off | 1500 | 1500 | 1500 | 0 | 0..1499 | 1.17e-15 | 1.75e-11 | all pass |

Simulator warnings (from successful batches; counts per message):

- nmos_on: Warning: toxe, toxp and dtox all given and toxe != toxp + dtox; dtox ignored. ×1500
- nmos_off: Warning: toxe, toxp and dtox all given and toxe != toxp + dtox; dtox ignored. ×1500
- pmos_on: Warning: toxe, toxp and dtox all given and toxe != toxp + dtox; dtox ignored. ×1500
- pmos_off: Warning: toxe, toxp and dtox all given and toxe != toxp + dtox; dtox ignored. ×1500

**Totals:** 6000 simulated, 6000 succeeded, 0 failed (see `failures.csv`).
**Validation:** all checks passed.
