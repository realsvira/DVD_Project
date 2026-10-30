# Part A2b: improving the transistor models near Vds ≈ 0 and the rails

## Why
In Part A2, v1 ON-state models had a 15–23 % median error on Id/Is at |Vds| < 0.05 V. Inside a static gate, ON transistors sit exactly there, and OFF transistors sit with their terminals on the rails.

## What changed (v2)
1. **Feature:** a signed log of Vds, `asinh(Vds / 1 µV)`. Near Vds = 0 the drain current is about G·Vds, so in the model's asinh(I) target space it varies like asinh(Vds). The network can use this feature almost linearly.
2. **Supplementary training data** (`data/transistor_aug_on`: 4,000 per ON case; `data/transistor_aug_off`: 2,000 per OFF case). It uses a targeted profile:
   - 40 % near-equal Vd and Vs, with |Vds| log-uniform between 1 nV and 0.05·VDD;
   - 40 % both terminals near a rail;
   - 20 % one terminal near a rail.

   VDD, T, W and the process parameters are drawn as in the spec. The data is split 80/20 into train/validation with a fixed seed. **The 10,000-row spec datasets are unchanged.**
3. **Model:** MLP only, because it beat the other three algorithms by 3–5× in A2. Grid: (128,128,128) with α = 1e-7 or 1e-5, and (192,192,192) with α = 1e-7. Trained on the original 70 % training split plus 80 % of the targeted samples, with 2 BLAS threads per fit.
4. **Selection (validation only, criterion declared in advance):**
   - score = mean of [original-val score, targeted-val score];
   - each score = mean over the 4 terminals of the median relative error;
   - the better of v1 and v2 is kept for each case.
5. **Fresh test sets,** used once to evaluate v1 and v2. The original A2 test split had already been used.
   - `data/transistor_test2_uniform`: 1,500 per case, spec distribution, new seed.
   - `data/transistor_test2_targeted`: 1,000 per case, targeted profile, new seed.

All supplementary simulations succeeded: 22,000 of 22,000.

## Validation (mean over Id/Ig/Is/Ib; median relative error)

| Case | v1 orig-val | v1 targeted-val | best v2 orig-val | best v2 targeted-val | Selected |
|---|---|---|---|---|---|
| NMOS ON | 2.25 % | 28.3 % | 3.51 % | 7.0 % | **v2** |
| NMOS OFF | 1.55 % | 2.89 % | 2.03 % | 2.83 % | v1 |
| PMOS ON | 2.57 % | 44.0 % | 3.58 % | 7.6 % | **v2** |
| PMOS OFF | 1.15 % | 2.11 % | 1.72 % | 2.47 % | v1 |

At |Vds| < 0.05 V (validation), the Id/Is median error for the ON models drops from about 100 % (v1) to about 23 % (v2).

## Fresh test sets (one-time; `results/a2b/test2_results.md`)

| Case | Model | Uniform test median / p95 | Targeted test median / p95 |
|---|---|---|---|
| NMOS ON | v1 | 2.28 % / 12.0 % | 25.6 % / 125,075 % |
| NMOS ON | **v2 (final)** | 3.53 % / 14.6 % | **7.24 % / 131 %** |
| NMOS OFF | **v1 (final)** | 1.49 % / 10.0 % | 3.07 % / 496 % |
| NMOS OFF | v2 | 2.07 % / 10.7 % | 3.06 % / 34 % |
| PMOS ON | v1 | 2.52 % / 16.3 % | 35.4 % / 65,413 % |
| PMOS ON | **v2 (final)** | 3.52 % / 14.2 % | **7.07 % / 77 %** |
| PMOS OFF | **v1 (final)** | 1.16 % / 7.3 % | 2.21 % / 448 % |
| PMOS OFF | v2 | 1.82 % / 10.4 % | 2.64 % / 36 % |

R²_t (uniform test) for the final models: NMOS ON 0.9999, NMOS OFF 0.9986, PMOS ON 0.9998, PMOS OFF 0.9974.

## Findings
- v1 ON models are **unusable at small |Vds|**. Their Id/Is predictions there are off by orders of magnitude (p95 above 10⁴ %). v2 fixes this: about 7 % median and below 130 % p95 on the targeted test. It costs about 1.2 percentage points of median error on the spec distribution.
- For OFF cases, v1 has the best medians but **long tails at rail biases**: targeted p95 is about 450–500 % for v1 versus 34–36 % for v2. The validation split shows the same thing (p95 3.6 / 5.4 for v1 versus 0.30 / 0.40 for v2). The pre-declared, median-only criterion kept v1. A criterion that also weighted the tails would have chosen v2 for OFF, using validation data alone.
- **99 % target:** still not reached as a 4-terminal mean on the spec distribution. The best case is PMOS OFF at 98.8 %.
- Gate and body currents remain the most accurate terminals (≤ 1–2 % median).

## Reproduce
```powershell
.\.venv\Scripts\python.exe scripts\a1_generate_datasets.py --skip-smoke --profile targeted --seed-salt 1 --n 4000 --cases nmos_on pmos_on --out data\transistor_aug_on
.\.venv\Scripts\python.exe scripts\a1_generate_datasets.py --skip-smoke --profile targeted --seed-salt 1 --n 2000 --cases nmos_off pmos_off --out data\transistor_aug_off
.\.venv\Scripts\python.exe scripts\a1_generate_datasets.py --skip-smoke --profile uniform  --seed-salt 2 --n 1500 --out data\transistor_test2_uniform
.\.venv\Scripts\python.exe scripts\a1_generate_datasets.py --skip-smoke --profile targeted --seed-salt 3 --n 1000 --out data\transistor_test2_targeted
.\.venv\Scripts\python.exe scripts\a2b_train_v2.py          # ~3 h on 12 cores (12 MLP fits)
.\.venv\Scripts\python.exe scripts\a2b_evaluate_test2.py    # one-time
```
Models: `trained_models/<case>.joblib` (final), `trained_models/v1/`, `trained_models/v2/`.
