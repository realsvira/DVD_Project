# Part A2: validation comparison

Generated 2026-11-13T01:50:21. Train/val only; the test split was not used.

Split sizes per case: {'test': 1500, 'train': 7000, 'val': 1500}

## Transform study (LightGBM, fixed settings)

Value = validation mean over the four terminals of the median relative error.

| transform | nmos_off | nmos_on | pmos_off | pmos_on | mean_over_cases |
|---|---|---|---|---|---|
| sign+log10\|I\| | 0.06891 | 0.08112 | 0.05519 | 0.08493 | 0.07254 |
| asinh(I/1e-15) | 0.07818 | 0.08681 | 0.05476 | 0.08776 | 0.07688 |
| asinh(I/1e-18) | 0.08979 | 0.09386 | 0.05921 | 0.09303 | 0.08397 |
| asinh(I/1e-12) | 0.09571 | 0.1093 | 0.07432 | 0.1152 | 0.09865 |
| asinh(I/1e-09) | 0.2173 | 0.2798 | 0.1346 | 0.2931 | 0.2312 |
| linear | 0.3326 | 0.3471 | 0.151 | 0.3379 | 0.2921 |

**Chosen transform: asinh(I/1e-15)**

## Four algorithms x four cases (best configuration of each algorithm)

R2_t = R^2 of asinh(I/1 fA); med/p95 = median/95th-percentile relative error |I_pred - I_spice| / max(|I_spice|, 1 fA).

| case | algo | avg_r2_t | avg_med_rel | avg_p95_rel | id_r2_t | id_med_rel | id_p95_rel | ig_r2_t | ig_med_rel | ig_p95_rel | is_r2_t | is_med_rel | is_p95_rel | ib_r2_t | ib_med_rel | ib_p95_rel | fit_seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nmos_off | mlp | 0.9973 | 0.01546 | 0.09418 | 0.997 | 0.02294 | 0.1572 | 0.9998 | 0.007262 | 0.02417 | 0.9943 | 0.02339 | 0.1447 | 0.9982 | 0.008262 | 0.05071 | 1036 |
| nmos_off | lightgbm | 0.9906 | 0.07152 | 0.5526 | 0.9845 | 0.1065 | 0.9703 | 0.9975 | 0.04777 | 0.1606 | 0.9829 | 0.1052 | 0.9418 | 0.9975 | 0.02664 | 0.1378 | 188.5 |
| nmos_off | random_forest | 0.9685 | 0.2199 | 1.224 | 0.9716 | 0.3022 | 1.65 | 0.9508 | 0.217 | 1.172 | 0.9729 | 0.3025 | 1.569 | 0.9788 | 0.05808 | 0.5043 | 32.42 |
| nmos_off | poly_ridge | 0.9416 | 0.466 | 3.19 | 0.8884 | 0.8538 | 6.338 | 0.9984 | 0.02205 | 0.08101 | 0.9009 | 0.8668 | 5.929 | 0.9788 | 0.1215 | 0.4107 | 1.29 |
| nmos_on | mlp | 0.9982 | 0.02255 | 0.1262 | 0.9966 | 0.03301 | 0.1953 | 0.9998 | 0.01138 | 0.06314 | 0.9966 | 0.03321 | 0.201 | 0.9999 | 0.01258 | 0.04539 | 991.4 |
| nmos_on | lightgbm | 0.9971 | 0.07941 | 0.4267 | 0.9967 | 0.07863 | 0.617 | 0.9984 | 0.06652 | 0.2045 | 0.9967 | 0.08054 | 0.5623 | 0.9966 | 0.09193 | 0.3229 | 42.55 |
| nmos_on | random_forest | 0.9876 | 0.251 | 1.863 | 0.999 | 0.2667 | 2.732 | 0.9804 | 0.2168 | 0.8706 | 0.999 | 0.2674 | 2.733 | 0.9721 | 0.253 | 1.115 | 19.3 |
| nmos_on | poly_ridge | 0.9606 | 0.5683 | 81.27 | 0.9288 | 1 | 162.4 | 0.9947 | 0.09584 | 0.3317 | 0.9288 | 1 | 161.9 | 0.9902 | 0.1773 | 0.49 | 1.57 |
| pmos_off | mlp | 0.9988 | 0.0115 | 0.0702 | 0.9987 | 0.01531 | 0.1002 | 0.9999 | 0.005433 | 0.02293 | 0.9971 | 0.01699 | 0.1151 | 0.9995 | 0.008257 | 0.04252 | 566.1 |
| pmos_off | lightgbm | 0.9817 | 0.05044 | 0.5498 | 0.9722 | 0.07123 | 0.9894 | 0.9981 | 0.0409 | 0.1396 | 0.9572 | 0.07735 | 0.9945 | 0.9993 | 0.01229 | 0.07562 | 72.47 |
| pmos_off | random_forest | 0.9551 | 0.1807 | 0.8503 | 0.9491 | 0.229 | 1.013 | 0.9507 | 0.1878 | 0.8756 | 0.9396 | 0.2412 | 1.013 | 0.9809 | 0.06486 | 0.4987 | 35.39 |
| pmos_off | poly_ridge | 0.9367 | 0.3839 | 2.724 | 0.8933 | 0.6927 | 5.122 | 0.9988 | 0.02354 | 0.08934 | 0.871 | 0.6946 | 5.262 | 0.9838 | 0.1249 | 0.422 | 1.17 |
| pmos_on | mlp | 0.9973 | 0.02569 | 0.1652 | 0.9949 | 0.04114 | 0.2736 | 0.9997 | 0.01041 | 0.05963 | 0.9947 | 0.03858 | 0.2778 | 0.9999 | 0.01264 | 0.04984 | 1029 |
| pmos_on | lightgbm | 0.9976 | 0.07927 | 0.4407 | 0.9982 | 0.08683 | 0.6153 | 0.9985 | 0.0497 | 0.1852 | 0.9979 | 0.0838 | 0.6012 | 0.9959 | 0.09676 | 0.3612 | 51.35 |
| pmos_on | random_forest | 0.9873 | 0.2604 | 1.957 | 0.9985 | 0.2994 | 3.023 | 0.98 | 0.1868 | 0.7888 | 0.9982 | 0.299 | 2.962 | 0.9724 | 0.2565 | 1.053 | 29.28 |
| pmos_on | poly_ridge | 0.9614 | 0.5643 | 52.9 | 0.9314 | 1 | 101.2 | 0.9939 | 0.06478 | 0.2919 | 0.9313 | 1 | 109.5 | 0.9892 | 0.1924 | 0.5583 | 1.2 |

## Selected models (lowest validation score)

| case | algo | params | score |
|---|---|---|---|
| nmos_off | mlp | {"hidden": [128, 128, 128], "alpha": 1e-05, "activation": "tanh", "solver": "lbfgs", "max_iter": 3000} | 0.01546 |
| nmos_on | mlp | {"hidden": [128, 128, 128], "alpha": 1e-07, "activation": "tanh", "solver": "lbfgs", "max_iter": 3000} | 0.02255 |
| pmos_off | mlp | {"hidden": [128, 128, 128], "alpha": 1e-07, "activation": "tanh", "solver": "lbfgs", "max_iter": 3000} | 0.0115 |
| pmos_on | mlp | {"hidden": [128, 128, 128], "alpha": 1e-07, "activation": "tanh", "solver": "lbfgs", "max_iter": 3000} | 0.02569 |

## Best hyperparameters per algorithm

| case | algo | params |
|---|---|---|
| nmos_off | mlp | {"hidden": [128, 128, 128], "alpha": 1e-05, "activation": "tanh", "solver": "lbfgs", "max_iter": 3000} |
| nmos_off | lightgbm | {"n_estimators": 4000, "learning_rate": 0.03, "num_leaves": 63} |
| nmos_off | random_forest | {"n_estimators": 300, "max_features": 1.0, "min_samples_leaf": 1} |
| nmos_off | poly_ridge | {"degree": 3, "alpha": 0.01} |
| nmos_on | mlp | {"hidden": [128, 128, 128], "alpha": 1e-07, "activation": "tanh", "solver": "lbfgs", "max_iter": 3000} |
| nmos_on | lightgbm | {"n_estimators": 4000, "learning_rate": 0.03, "num_leaves": 15} |
| nmos_on | random_forest | {"n_estimators": 300, "max_features": 1.0, "min_samples_leaf": 1} |
| nmos_on | poly_ridge | {"degree": 3, "alpha": 0.01} |
| pmos_off | mlp | {"hidden": [128, 128, 128], "alpha": 1e-07, "activation": "tanh", "solver": "lbfgs", "max_iter": 3000} |
| pmos_off | lightgbm | {"n_estimators": 4000, "learning_rate": 0.03, "num_leaves": 15} |
| pmos_off | random_forest | {"n_estimators": 300, "max_features": 1.0, "min_samples_leaf": 1} |
| pmos_off | poly_ridge | {"degree": 3, "alpha": 0.01} |
| pmos_on | mlp | {"hidden": [128, 128, 128], "alpha": 1e-07, "activation": "tanh", "solver": "lbfgs", "max_iter": 3000} |
| pmos_on | lightgbm | {"n_estimators": 4000, "learning_rate": 0.03, "num_leaves": 15} |
| pmos_on | random_forest | {"n_estimators": 300, "max_features": 1.0, "min_samples_leaf": 1} |
| pmos_on | poly_ridge | {"degree": 3, "alpha": 0.01} |
