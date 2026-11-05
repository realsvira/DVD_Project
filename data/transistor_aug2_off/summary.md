# Part A1 dataset generation summary

- Created: 2026-11-13T15:24:25  |  command: `a1_generate_datasets.py --skip-smoke --no-progress --workers 6 --profile targeted --seed-salt 6 --n 2000 --cases nmos_off pmos_off --out data/transistor_aug2_off`
- Model `models\45nm_HP.pm` (sha256 9ab66a7bc4a5...), ngspice 44.2
- Git commit 2188028979 (dirty)
- Output folder: `data\transistor_aug2_off`
- This run: 4000 simulations in 37.7 s (106 sim/s)

| Case | Rows | Simulated | Succeeded | Failed | Sample ids | max abs KCL sum (A) | median abs Id (A) | Checks |
|---|---:|---:|---:|---:|---|---:|---:|---|
| nmos_off | 2000 | 2000 | 2000 | 0 | 0..1999 | 9.04e-23 | 1.50e-11 | all pass |
| pmos_off | 2000 | 2000 | 2000 | 0 | 0..1999 | 1.31e-15 | 4.24e-11 | all pass |

Simulator warnings (from successful batches; counts per message):

- nmos_off: Warning: toxe, toxp and dtox all given and toxe != toxp + dtox; dtox ignored. ×2000
- pmos_off: Warning: toxe, toxp and dtox all given and toxe != toxp + dtox; dtox ignored. ×2000

**Totals:** 4000 simulated, 4000 succeeded, 0 failed (see `failures.csv`).
**Validation:** all checks passed.
