# Part A2: validation comparison

Generated 2026-11-13T18:46:18. Train/val only; the test split was not used.

Split sizes per case: {'test': 1500, 'train': 7000, 'val': 1500}

## Transform study (LightGBM, fixed settings)

Value = validation mean over the four terminals of the median relative error.

| transform | nmos_off | nmos_on | pmos_off | pmos_on | mean_over_cases |
|---|---|---|---|---|---|
| sign+log10\|I\| | 0.06891 | 0.08112 | 0.05519 | 0.08493 | 0.07254 |
| asinh(I/1e-15) | 0.07818 | 0.08681 | 0.05476 | 0.08776 | 0.07688 |
| asinh(I/1e-18) | 0.08979 | 0.09386 | 0.05921 | 0.09303 | 0.08397 |
| linear | 0.3326 | 0.3471 | 0.151 | 0.3379 | 0.2921 |

**Chosen transform: asinh(I/1e-15)**

## Four algorithms x four cases (best configuration of each algorithm)

R2_t = R^2 of asinh(I/1 fA); med/p95 = median/95th-percentile relative error |I_pred - I_spice| / max(|I_spice|, 1 fA).

| case | algo | avg_r2_t | avg_med_rel | avg_p95_rel | id_r2_t | id_med_rel | id_p95_rel | ig_r2_t | ig_med_rel | ig_p95_rel | is_r2_t | is_med_rel | is_p95_rel | ib_r2_t | ib_med_rel | ib_p95_rel | fit_seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nmos_off | lightgbm | 0.9892 | 0.1056 | 0.5926 | 0.9816 | 0.1839 | 0.9961 | 0.9959 | 0.06752 | 0.2145 | 0.9819 | 0.1498 | 1.001 | 0.9973 | 0.02132 | 0.1593 | 5.59 |
| nmos_off | mlp | 0.991 | 0.1354 | 0.6136 | 0.9905 | 0.209 | 1.055 | 0.9977 | 0.04313 | 0.1502 | 0.988 | 0.2205 | 0.9988 | 0.9879 | 0.06903 | 0.2505 | 7.91 |
| nmos_off | random_forest | 0.967 | 0.2298 | 1.173 | 0.9712 | 0.3146 | 1.54 | 0.9489 | 0.2238 | 1.095 | 0.9715 | 0.3055 | 1.528 | 0.9765 | 0.07522 | 0.53 | 2.16 |
| nmos_off | poly_ridge | 0.9103 | 0.5716 | 5.997 | 0.8409 | 0.9741 | 13.28 | 0.9958 | 0.04879 | 0.1429 | 0.8542 | 0.9747 | 9.665 | 0.9505 | 0.2888 | 0.8932 | 0.24 |
| nmos_on | lightgbm | 0.996 | 0.1087 | 0.5146 | 0.9972 | 0.1053 | 0.6518 | 0.9964 | 0.09637 | 0.3146 | 0.9971 | 0.1041 | 0.6533 | 0.9932 | 0.129 | 0.4388 | 5.53 |
| nmos_on | mlp | 0.9965 | 0.1842 | 0.9538 | 0.9962 | 0.2842 | 1.689 | 0.9971 | 0.06967 | 0.2509 | 0.9962 | 0.2855 | 1.564 | 0.9965 | 0.09728 | 0.3109 | 8.37 |
| nmos_on | random_forest | 0.9861 | 0.2659 | 1.698 | 0.9991 | 0.2702 | 2.35 | 0.9763 | 0.2447 | 0.9533 | 0.9992 | 0.2716 | 2.35 | 0.9699 | 0.2773 | 1.139 | 2.09 |
| nmos_on | poly_ridge | 0.9324 | 0.6063 | 165.5 | 0.8824 | 1 | 330.1 | 0.9924 | 0.1164 | 0.3797 | 0.8824 | 1 | 330.4 | 0.9723 | 0.3089 | 1.023 | 0.32 |
| pmos_off | lightgbm | 0.9819 | 0.08411 | 0.59 | 0.9719 | 0.1174 | 0.9948 | 0.9957 | 0.06692 | 0.2186 | 0.9616 | 0.1269 | 0.996 | 0.9983 | 0.02522 | 0.1504 | 3.28 |
| pmos_off | mlp | 0.9931 | 0.08788 | 0.5687 | 0.9919 | 0.1266 | 0.9475 | 0.9983 | 0.03621 | 0.1356 | 0.9878 | 0.153 | 0.9799 | 0.9943 | 0.0357 | 0.2116 | 5.99 |
| pmos_off | random_forest | 0.9533 | 0.1917 | 0.8692 | 0.9474 | 0.2509 | 1.005 | 0.9492 | 0.1962 | 0.9289 | 0.9379 | 0.2399 | 1.042 | 0.9789 | 0.07977 | 0.5008 | 2.72 |
| pmos_off | poly_ridge | 0.8906 | 0.5759 | 4.592 | 0.8239 | 0.9785 | 10.37 | 0.996 | 0.04985 | 0.1556 | 0.7921 | 0.9724 | 6.955 | 0.9504 | 0.3026 | 0.8898 | 0.2 |
| pmos_on | lightgbm | 0.9963 | 0.1131 | 0.5337 | 0.9982 | 0.1194 | 0.6748 | 0.9968 | 0.0799 | 0.2718 | 0.998 | 0.126 | 0.7315 | 0.9924 | 0.127 | 0.4568 | 5.68 |
| pmos_on | mlp | 0.9957 | 0.1958 | 0.874 | 0.9956 | 0.3167 | 1.539 | 0.9955 | 0.06054 | 0.2635 | 0.995 | 0.3114 | 1.378 | 0.9966 | 0.0943 | 0.3154 | 6.93 |
| pmos_on | random_forest | 0.9858 | 0.2692 | 1.979 | 0.9986 | 0.301 | 2.983 | 0.9767 | 0.2068 | 0.9238 | 0.9982 | 0.3003 | 2.89 | 0.9696 | 0.2688 | 1.119 | 2.45 |
| pmos_on | poly_ridge | 0.9324 | 0.6003 | 200.9 | 0.8856 | 1 | 408.7 | 0.9923 | 0.06522 | 0.3014 | 0.8858 | 1 | 393.3 | 0.9659 | 0.336 | 1.2 | 0.11 |

## Selected models (lowest validation score)

| case | algo | params | score |
|---|---|---|---|
| nmos_off | lightgbm | {"n_estimators": 200, "learning_rate": 0.1, "num_leaves": 31} | 0.1056 |
| nmos_on | lightgbm | {"n_estimators": 200, "learning_rate": 0.1, "num_leaves": 31} | 0.1087 |
| pmos_off | lightgbm | {"n_estimators": 200, "learning_rate": 0.1, "num_leaves": 31} | 0.08411 |
| pmos_on | lightgbm | {"n_estimators": 200, "learning_rate": 0.1, "num_leaves": 31} | 0.1131 |

## Best hyperparameters per algorithm

| case | algo | params |
|---|---|---|
| nmos_off | lightgbm | {"n_estimators": 200, "learning_rate": 0.1, "num_leaves": 31} |
| nmos_off | mlp | {"hidden": [32, 32], "alpha": 1e-05, "activation": "tanh", "solver": "lbfgs", "max_iter": 200} |
| nmos_off | random_forest | {"n_estimators": 50, "max_features": 0.66, "min_samples_leaf": 1} |
| nmos_off | poly_ridge | {"degree": 2, "alpha": 1e-05} |
| nmos_on | lightgbm | {"n_estimators": 200, "learning_rate": 0.1, "num_leaves": 31} |
| nmos_on | mlp | {"hidden": [32, 32], "alpha": 1e-05, "activation": "tanh", "solver": "lbfgs", "max_iter": 200} |
| nmos_on | random_forest | {"n_estimators": 50, "max_features": 0.66, "min_samples_leaf": 1} |
| nmos_on | poly_ridge | {"degree": 2, "alpha": 1e-05} |
| pmos_off | lightgbm | {"n_estimators": 200, "learning_rate": 0.1, "num_leaves": 31} |
| pmos_off | mlp | {"hidden": [32, 32], "alpha": 1e-05, "activation": "tanh", "solver": "lbfgs", "max_iter": 200} |
| pmos_off | random_forest | {"n_estimators": 50, "max_features": 0.66, "min_samples_leaf": 1} |
| pmos_off | poly_ridge | {"degree": 2, "alpha": 1e-05} |
| pmos_on | lightgbm | {"n_estimators": 200, "learning_rate": 0.1, "num_leaves": 31} |
| pmos_on | mlp | {"hidden": [32, 32], "alpha": 1e-05, "activation": "tanh", "solver": "lbfgs", "max_iter": 200} |
| pmos_on | random_forest | {"n_estimators": 50, "max_features": 0.66, "min_samples_leaf": 1} |
| pmos_on | poly_ridge | {"degree": 2, "alpha": 1e-05} |
