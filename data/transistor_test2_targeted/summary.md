# Part A1 dataset generation summary

- Created: 2026-11-13T08:40:19  |  command: `a1_generate_datasets.py --skip-smoke --no-progress --profile targeted --seed-salt 3 --n 1000 --out data/transistor_test2_targeted`
- Model `models\45nm_HP.pm` (sha256 9ab66a7bc4a5...), ngspice 44.2
- Git commit e5fee494f7 (dirty)
- Output folder: `data\transistor_test2_targeted`
- This run: 4000 simulations in 34.8 s (115 sim/s)

| Case | Rows | Simulated | Succeeded | Failed | Sample ids | max abs KCL sum (A) | median abs Id (A) | Checks |
|---|---:|---:|---:|---:|---|---:|---:|---|
| nmos_on | 1000 | 1000 | 1000 | 0 | 0..999 | 9.22e-16 | 4.29e-08 | all pass |
| nmos_off | 1000 | 1000 | 1000 | 0 | 0..999 | 4.65e-23 | 1.54e-11 | all pass |
| pmos_on | 1000 | 1000 | 1000 | 0 | 0..999 | 1.39e-16 | 9.14e-09 | all pass |
| pmos_off | 1000 | 1000 | 1000 | 0 | 0..999 | 1.91e-15 | 5.45e-11 | all pass |

Simulator warnings (from successful batches; counts per message):

- nmos_on: Warning: toxe, toxp and dtox all given and toxe != toxp + dtox; dtox ignored. ×1000
- nmos_off: Warning: toxe, toxp and dtox all given and toxe != toxp + dtox; dtox ignored. ×1000
- pmos_on: Warning: toxe, toxp and dtox all given and toxe != toxp + dtox; dtox ignored. ×1000
- pmos_off: Warning: toxe, toxp and dtox all given and toxe != toxp + dtox; dtox ignored. ×1000

**Totals:** 4000 simulated, 4000 succeeded, 0 failed (see `failures.csv`).
**Validation:** all checks passed.
