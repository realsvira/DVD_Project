# Part A1 dataset generation summary

- Created: 2026-11-13T15:23:43  |  command: `a1_generate_datasets.py --skip-smoke --no-progress --workers 6 --profile targeted --seed-salt 6 --n 4000 --cases nmos_on pmos_on --out data/transistor_aug2_on`
- Model `models\45nm_HP.pm` (sha256 9ab66a7bc4a5...), ngspice 44.2
- Git commit 2188028979 (dirty)
- Output folder: `data\transistor_aug2_on`
- This run: 8000 simulations in 84.0 s (95 sim/s)

| Case | Rows | Simulated | Succeeded | Failed | Sample ids | max abs KCL sum (A) | median abs Id (A) | Checks |
|---|---:|---:|---:|---:|---|---:|---:|---|
| nmos_on | 4000 | 4000 | 4000 | 0 | 0..3999 | 1.38e-15 | 2.76e-08 | all pass |
| pmos_on | 4000 | 4000 | 4000 | 0 | 0..3999 | 2.00e-16 | 2.05e-08 | all pass |

Simulator warnings (from successful batches; counts per message):

- nmos_on: Warning: toxe, toxp and dtox all given and toxe != toxp + dtox; dtox ignored. ×4000
- pmos_on: Warning: toxe, toxp and dtox all given and toxe != toxp + dtox; dtox ignored. ×4000

**Totals:** 8000 simulated, 8000 succeeded, 0 failed (see `failures.csv`).
**Validation:** all checks passed.
