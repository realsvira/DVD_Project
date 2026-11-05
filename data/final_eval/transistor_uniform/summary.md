# Part A1 dataset generation summary

- Created: 2026-11-13T13:40:35  |  command: `a1_generate_datasets.py --skip-smoke --no-progress --profile uniform --seed-salt 4 --n 1500 --out data/final_eval/transistor_uniform --workers 8`
- Model `models\45nm_HP.pm` (sha256 9ab66a7bc4a5...), ngspice 44.2
- Git commit 2188028979 (dirty)
- Output folder: `data\final_eval\transistor_uniform`
- This run: 6000 simulations in 36.5 s (165 sim/s)

| Case | Rows | Simulated | Succeeded | Failed | Sample ids | max abs KCL sum (A) | median abs Id (A) | Checks |
|---|---:|---:|---:|---:|---|---:|---:|---|
| nmos_on | 1500 | 1500 | 1500 | 0 | 0..1499 | 1.01e-15 | 3.55e-05 | all pass |
| nmos_off | 1500 | 1500 | 1500 | 0 | 0..1499 | 1.72e-23 | 6.84e-12 | all pass |
| pmos_on | 1500 | 1500 | 1500 | 0 | 0..1499 | 1.62e-16 | 1.89e-05 | all pass |
| pmos_off | 1500 | 1500 | 1500 | 0 | 0..1499 | 1.47e-15 | 1.76e-11 | all pass |

Simulator warnings (from successful batches; counts per message):

- nmos_on: Warning: toxe, toxp and dtox all given and toxe != toxp + dtox; dtox ignored. ×1500
- nmos_off: Warning: toxe, toxp and dtox all given and toxe != toxp + dtox; dtox ignored. ×1500
- pmos_on: Warning: toxe, toxp and dtox all given and toxe != toxp + dtox; dtox ignored. ×1500
- pmos_off: Warning: toxe, toxp and dtox all given and toxe != toxp + dtox; dtox ignored. ×1500

**Totals:** 6000 simulated, 6000 succeeded, 0 failed (see `failures.csv`).
**Validation:** all checks passed.
