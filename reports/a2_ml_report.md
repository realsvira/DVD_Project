# Part A2: ML transistor models (report)

Datasets: `data/transistor/<case>.csv`, which holds 4 × 10,000 SPICE samples (45 nm HP, ngspice 44.2).

There is one model per case: NMOS ON, NMOS OFF, PMOS ON and PMOS OFF. Each one predicts all four terminal currents (Id, Ig, Is, Ib). Currents are in A and are **positive when they flow into the device**. Signs are kept throughout.

All numbers below are measured. Sources:
- `results/a2/validation_comparison.csv`
- `results/a2/test_results.csv`
- `results/a2/test_error_analysis.csv`

## 1. Setup

| Item | Choice |
|---|---|
| Split | 70 / 15 / 15 (7,000 / 1,500 / 1,500 per case), fixed seed, saved in `data/transistor/splits.csv` |
| Leakage control | Test rows are removed before any fitting. Feature and target standardisation are fit on the training split. Tuning, transform choice and model selection use validation only. The test set is evaluated once by `a2_evaluate_test.py`, which refuses to run a second time. |
| Features (17) | VDD, T (K), log10 W, Vd, Vs, Vds, \|Vds\|, Vgs, Vgd, Vbs, Vbd, source offset from its rail, vth0/toxe/ndep/u0/xj relative to nominal |
| Targets | id, ig, is, ib together (multi-output), transformed with **asinh(I / 1e-15 A)** and then standardised |
| Algorithms | polynomial ridge, random forest, LightGBM, MLP (L-BFGS, tanh) |
| Tuning | grid search on validation: 6 / 6 / 8 / 8 configurations per case (112 fits) |
| Selection | lowest validation mean over the 4 terminals of the median relative error |
| Reproducibility | fixed seeds; every fit pinned to 2 BLAS threads. The refit models reproduce the tuned validation scores exactly. |

**Error definitions:**
- **Relative error:** |I_pred − I_spice| / max(|I_spice|, 1 fA). The 1 fA floor stops sub-femtoamp currents from producing meaningless relative errors.
- **Absolute error:** in A.
- **Scale error:** the error divided by the sample's largest terminal current.
- **Sign accuracy:** counted only for samples with |I| ≥ 10 fA.
- **R²_t:** R² of asinh(I/1 fA). This is a common space for every model.
- **R²_raw:** R² on the currents in A.
- **Accuracy:** 100 % − relative error, where quoted.

### Target-transform study
This used LightGBM with fixed settings. The score is the validation mean of the median relative error.

| Transform | NMOS OFF | NMOS ON | PMOS OFF | PMOS ON | Mean |
|---|---|---|---|---|---|
| sign classifier + log10\|I\| | 0.069 | 0.081 | 0.055 | 0.085 | **0.073** |
| asinh(I/1e-15) | 0.078 | 0.087 | 0.055 | 0.088 | 0.077 |
| asinh(I/1e-18) | 0.090 | 0.094 | 0.059 | 0.093 | 0.084 |
| asinh(I/1e-12) | 0.096 | 0.109 | 0.074 | 0.115 | 0.099 |
| asinh(I/1e-09) | 0.217 | 0.280 | 0.135 | 0.293 | 0.231 |
| linear (I) | 0.333 | 0.347 | 0.151 | 0.338 | 0.292 |

Linear targets fail because the currents span about 11 decades. Sign + log was 0.004 better, but it needs a separate sign classifier per terminal, so asinh(I/1e-15) was chosen as the best transform that each algorithm can use within a single model.

## 2. Validation comparison (4 algorithms × 4 cases)
Each algorithm is shown with its best configuration. Values are means over Id/Ig/Is/Ib.

| Case | Algorithm | R²_t | Median rel. err | p95 rel. err |
|---|---|---|---|---|
| NMOS ON | **MLP** | 0.9982 | **2.25 %** | 12.6 % |
| NMOS ON | LightGBM | 0.9971 | 7.94 % | 42.7 % |
| NMOS ON | Random forest | 0.9876 | 25.1 % | 186 % |
| NMOS ON | Poly ridge | 0.9606 | 56.8 % | 8127 % |
| NMOS OFF | **MLP** | 0.9973 | **1.55 %** | 9.4 % |
| NMOS OFF | LightGBM | 0.9906 | 7.15 % | 55.3 % |
| NMOS OFF | Random forest | 0.9685 | 22.0 % | 122 % |
| NMOS OFF | Poly ridge | 0.9416 | 46.6 % | 319 % |
| PMOS ON | **MLP** | 0.9973 | **2.57 %** | 16.5 % |
| PMOS ON | LightGBM | 0.9976 | 7.93 % | 44.1 % |
| PMOS ON | Random forest | 0.9873 | 26.0 % | 196 % |
| PMOS ON | Poly ridge | 0.9614 | 56.4 % | 5290 % |
| PMOS OFF | **MLP** | 0.9988 | **1.15 %** | 7.0 % |
| PMOS OFF | LightGBM | 0.9817 | 5.04 % | 55.0 % |
| PMOS OFF | Random forest | 0.9551 | 18.1 % | 85.0 % |
| PMOS OFF | Poly ridge | 0.9367 | 38.4 % | 272 % |

Per-terminal values are in `results/a2/validation_comparison.csv`.

**Selected:** an MLP with 3 hidden layers of 128 tanh units, trained with L-BFGS for up to 3000 iterations, in every case. Its weight-decay penalty (alpha) is 1e-7, except NMOS OFF, which uses 1e-5.

**Why the MLP wins:** the currents are smooth, strongly nonlinear functions of the inputs (exponential in voltage and temperature, and in toxe). The MLP fits that as one smooth function. Tree models produce piecewise-constant approximations, which are coarse over 11 decades. Polynomial ridge cannot follow the exponentials and gets the Id sign wrong near zero crossings.

## 3. Test-set results (one-time evaluation, 1,500 samples per case)

| Case | Terminal | R²_t | Median rel. err | p95 rel. err | < 1 % | < 5 % | Median abs (A) | Sign acc. |
|---|---|---|---|---|---|---|---|---|
| NMOS ON | Id | 0.9980 | 3.63 % | 20.6 % | 14 % | 64 % | 8.2e-07 | 99.9 % |
| | Ig | 0.9997 | 1.18 % | 7.2 % | 44 % | 92 % | 5.2e-14 | 100 % |
| | Is | 0.9980 | 3.60 % | 20.2 % | 16 % | 64 % | 7.9e-07 | 99.9 % |
| | Ib | 0.9999 | 1.31 % | 4.9 % | 40 % | 95 % | 4.9e-15 | 100 % |
| NMOS OFF | Id | 0.9979 | 2.31 % | 15.7 % | 24 % | 77 % | 1.4e-13 | 99.9 % |
| | Ig | 0.9999 | 0.73 % | 2.5 % | 64 % | 99 % | 4.4e-14 | 100 % |
| | Is | 0.9988 | 2.28 % | 14.4 % | 22 % | 77 % | 1.5e-13 | 99.9 % |
| | Ib | 0.9994 | 0.83 % | 5.8 % | 57 % | 94 % | 1.2e-16 | 100 % |
| PMOS ON | Id | 0.9953 | 4.30 % | 33.2 % | 14 % | 56 % | 4.4e-07 | 99.7 % |
| | Ig | 0.9998 | 1.03 % | 6.6 % | 49 % | 93 % | 2.3e-13 | 100 % |
| | Is | 0.9950 | 4.37 % | 29.3 % | 13 % | 55 % | 4.8e-07 | 99.3 % |
| | Ib | 0.9999 | 1.29 % | 5.2 % | 39 % | 94 % | 3.5e-15 | 100 % |
| PMOS OFF | Id | 0.9992 | 1.50 % | 10.0 % | 36 % | 86 % | 2.4e-13 | 100 % |
| | Ig | 0.9999 | 0.57 % | 2.1 % | 73 % | 99 % | 1.9e-13 | 100 % |
| | Is | 0.9927 | 1.69 % | 12.3 % | 32 % | 84 % | 2.9e-13 | 99.5 % |
| | Ib | 0.9987 | 0.82 % | 4.8 % | 58 % | 95 % | 1.2e-16 | 100 % |

**Mean over the four terminals** (test):

| Case | Median rel. err | p95 rel. err | R²_t | Accuracy (100 % − median) |
|---|---|---|---|---|
| NMOS ON | 2.43 % | 13.2 % | 0.9989 | 97.6 % |
| NMOS OFF | 1.54 % | 9.6 % | 0.9990 | 98.5 % |
| PMOS ON | 2.75 % | 18.6 % | 0.9975 | 97.3 % |
| PMOS OFF | 1.14 % | 7.3 % | 0.9976 | 98.9 % |

The test numbers agree with validation to within about 0.2 percentage points, so the selection did not overfit the validation split.

**99 % target: not reached as a mean over the four terminals in any case.** By the 100 % − median-relative-error definition, the best case is 98.9 % (PMOS OFF). It *is* reached by the gate and body currents in the OFF cases: Ig gives 99.3 % for NMOS and 99.4 % for PMOS, and Ib gives 99.2 % for both. The drain and source currents keep the mean below 99 %.

Plots: `results/a2/fig_pred_vs_spice_<case>.png` (four panels each; red points mark a predicted sign that differs from SPICE).

## 4. Error analysis

- **Ig and Ib** are learned best: median 0.6–1.3 %, R²_t ≥ 0.9987, signs 100 % correct. They are smooth, single-signed functions of the bias.
- **Id and Is** carry most of the error, in two regimes (`test_error_analysis.csv`):
  - **ON cases at small |Vds|.** The median error is 15–23 % for |Vds| < 0.05 V, against about 3 % for |Vds| > 0.2 V. Id passes through zero at Vds = 0, so a small absolute error is a large relative error there. **This matters for Part B:** inside a static gate, conducting transistors sit at Vds ≈ 0.
  - **OFF cases with |I| < 10 fA.** Here the channel current and the gate-to-drain tunnelling current nearly cancel (see the data-quality report). The relative error is large, but the absolute error is only about 1e-14 A.
- **Corners of the input space** (for example Vd = VDD with Vs = 0 exactly, the bias of a gate's OFF transistor) are sampled sparsely and are somewhat worse. At the nominal NMOS OFF point, the Id error is 8 %.
- Temperature has little effect on accuracy.
- The source offset matters mostly through the cancellation regime described above.
- Kirchhoff's law is not enforced by the model. The median |sum of predicted currents| / max|I| is recorded in `test_evaluation_record.json`.

## 5. Limitations and suggested improvements (before Part B)

1. **ON-state Id/Is near Vds = 0.** Options:
   - add training samples concentrated at small |Vds| and at the gate-relevant corners (Vd, Vs ∈ {0, VDD}), which the spec allows ("the number of samples may be increased");
   - or predict Id/Vds (a conductance) near zero.
2. **Sign + log10|I|** was slightly better than asinh in the LightGBM study. A sign head added to the MLP may help the Id/Is tails.
3. **Kirchhoff consistency:** predict three currents and derive the fourth, or add a penalty to training.
4. All models are trained on the 70 % training split only. Refitting the chosen settings on train + validation would add 15 % more data without touching the test set.

## 6. Reproduce

```powershell
.\.venv\Scripts\python.exe scripts\a2_train_models.py      # ~1.5 h; cached/resumable
.\.venv\Scripts\python.exe scripts\a2_evaluate_test.py     # one-time
.\.venv\Scripts\python.exe scripts\a2_predict.py --device nmos --vd 1.0 --vg 0 --vs 0 --vb 0 --vdd 1.0 --temp 27 --w 90e-9 --spice
```
Saved models: `trained_models/<case>.joblib`. Each is a `CurrentModel` object holding its preprocessing, feature list, nominal process values, hyperparameters, validation metrics and the split/dataset hashes.
