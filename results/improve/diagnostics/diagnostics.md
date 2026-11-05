# Improvement phase: diagnostics of the current final models (train/validation only)

## Optimisation

| case | hidden | alpha | n_iter | max_iter | hit_cap | train_loss |
|---|---|---|---|---|---|---|
| nmos_on | (128, 128, 128) | 1e-05 | 3000 | 3000 | True | 6.557e-05 |
| nmos_off | (128, 128, 128) | 1e-05 | 3000 | 3000 | True | 4.779e-05 |
| pmos_on | (128, 128, 128) | 1e-05 | 3000 | 3000 | True | 7.241e-05 |
| pmos_off | (192, 192, 192) | 1e-07 | 3000 | 3000 | True | 6.631e-05 |

Every final model stopped at the L-BFGS iteration cap: the fits are optimisation-limited.

## Train vs validation (mean over terminals, median relative error)

| case | train | val_orig | val_targeted |
|---|---|---|---|
| nmos_off | 0.02009 | 0.02028 | 0.0283 |
| nmos_on | 0.039 | 0.03505 | 0.07011 |
| pmos_off | 0.01728 | 0.01719 | 0.02467 |
| pmos_on | 0.03889 | 0.03578 | 0.07575 |

Train ≈ validation means the models are not over-fitting; the error floor is set by how well the network is fitted (under-fitting), not by too little data.

## Sign accuracy and tiny currents

| case | terminal | sign_acc | frac_below_10fA | share_of_p95_tail_below_10fA |
|---|---|---|---|---|
| nmos_on | id | 0.9871 | 0.02174 | 0.1391 |
| nmos_on | ig | 1 | 0.03391 | 0.3826 |
| nmos_on | is | 0.9893 | 0.02696 | 0.1304 |
| nmos_on | ib | 1 | 0 | 0 |
| nmos_off | id | 0.9995 | 0.006842 | 0.1053 |
| nmos_off | ig | 1 | 0.01947 | 0.2842 |
| nmos_off | is | 0.9968 | 0.01105 | 0.2 |
| nmos_off | ib | 1 | 0.1316 | 0.6 |
| pmos_on | id | 0.9951 | 0.0187 | 0.1565 |
| pmos_on | ig | 1 | 0.02565 | 0.2435 |
| pmos_on | is | 0.9947 | 0.0213 | 0.1217 |
| pmos_on | ib | 1 | 0 | 0 |
| pmos_off | id | 0.9979 | 0.01158 | 0.1895 |
| pmos_off | ig | 1 | 0.01526 | 0.2842 |
| pmos_off | is | 0.9995 | 0.01211 | 0.2105 |
| pmos_off | ib | 1 | 0.3226 | 0.6105 |

## Error vs distance to the nearest training sample (coverage)

| case | nn_distance_quartile | median_dist | med_rel_id | med_rel_ig | med_rel_is | med_rel_ib |
|---|---|---|---|---|---|---|
| nmos_on | closest 25% | 1.301 | 0.06354 | 0.01758 | 0.05806 | 0.01658 |
| nmos_on | 25-50% | 1.515 | 0.06498 | 0.01693 | 0.05942 | 0.01772 |
| nmos_on | 50-75% | 1.674 | 0.06703 | 0.01676 | 0.06479 | 0.01841 |
| nmos_on | farthest 25% | 1.866 | 0.08419 | 0.02022 | 0.08105 | 0.02092 |
| nmos_off | closest 25% | 1.331 | 0.02606 | 0.009044 | 0.03136 | 0.008741 |
| nmos_off | 25-50% | 1.549 | 0.02823 | 0.01111 | 0.02984 | 0.009658 |
| nmos_off | 50-75% | 1.708 | 0.03164 | 0.01154 | 0.0375 | 0.01033 |
| nmos_off | farthest 25% | 1.912 | 0.04031 | 0.01412 | 0.0389 | 0.01096 |
| pmos_on | closest 25% | 1.304 | 0.0582 | 0.01497 | 0.05806 | 0.01478 |
| pmos_on | 25-50% | 1.524 | 0.06237 | 0.01546 | 0.06693 | 0.0169 |
| pmos_on | 50-75% | 1.673 | 0.06936 | 0.01753 | 0.07386 | 0.01975 |
| pmos_on | farthest 25% | 1.882 | 0.07998 | 0.0201 | 0.07582 | 0.02248 |
| pmos_off | closest 25% | 1.328 | 0.02039 | 0.008449 | 0.0237 | 0.01288 |
| pmos_off | 25-50% | 1.551 | 0.0214 | 0.00981 | 0.02516 | 0.01418 |
| pmos_off | 50-75% | 1.717 | 0.02506 | 0.01079 | 0.02598 | 0.01276 |
| pmos_off | farthest 25% | 1.927 | 0.02784 | 0.01273 | 0.02773 | 0.01528 |

Full factor tables: `error_by_factor.csv`.
