# Part A1 dataset generation summary

- Created: 2026-11-13T13:41:02  |  command: `a1_generate_datasets.py --skip-smoke --no-progress --profile targeted --seed-salt 5 --n 1000 --out data/final_eval/transistor_targeted --workers 8`
- Model `models\45nm_HP.pm` (sha256 9ab66a7bc4a5...), ngspice 44.2
- Git commit 2188028979 (dirty)
- Output folder: `data\final_eval\transistor_targeted`
- This run: 4000 simulations in 25.5 s (157 sim/s)

| Case | Rows | Simulated | Succeeded | Failed | Sample ids | max abs KCL sum (A) | median abs Id (A) | Checks |
|---|---:|---:|---:|---:|---|---:|---:|---|
| nmos_on | 1000 | 1000 | 1000 | 0 | 0..999 | 1.40e-15 | 4.94e-08 | all pass |
| nmos_off | 1000 | 1000 | 1000 | 0 | 0..999 | 4.79e-23 | 1.62e-11 | all pass |
| pmos_on | 1000 | 1000 | 1000 | 0 | 0..999 | 1.66e-16 | 1.75e-08 | all pass |
| pmos_off | 1000 | 1000 | 1000 | 0 | 0..999 | 9.17e-16 | 4.76e-11 | all pass |

Simulator warnings (from successful batches; counts per message):

- nmos_on: Warning: toxe, toxp and dtox all given and toxe != toxp + dtox; dtox ignored. ×1000
- nmos_off: Warning: toxe, toxp and dtox all given and toxe != toxp + dtox; dtox ignored. ×1000
- pmos_on: Warning: toxe, toxp and dtox all given and toxe != toxp + dtox; dtox ignored. ×1000
- pmos_off: Warning: toxe, toxp and dtox all given and toxe != toxp + dtox; dtox ignored. ×1000

**Totals:** 4000 simulated, 4000 succeeded, 0 failed (see `failures.csv`).
**Validation:** all checks passed.
