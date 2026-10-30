# Part A2b: fresh test-set results (one-time evaluation)

Evaluated 2026-11-13T12:01:24. Both test sets were generated with independent seeds after Part A2 and were not used for training, tuning or selection.

## Mean over Id/Ig/Is/Ib

| case | model | final | test_set | med_rel | p95_rel | r2_t |
|---|---|---|---|---|---|---|
| nmos_on | v1 | False | uniform | 0.02277 | 0.12 | 0.9975 |
| nmos_on | v2 | True | uniform | 0.03526 | 0.1464 | 0.9999 |
| nmos_on | v1 | False | targeted | 0.2561 | 1251 | 0.8432 |
| nmos_on | v2 | True | targeted | 0.07238 | 1.313 | 0.9867 |
| nmos_off | v1 | True | uniform | 0.01494 | 0.1002 | 0.9986 |
| nmos_off | v2 | False | uniform | 0.02075 | 0.1069 | 0.9991 |
| nmos_off | v1 | True | targeted | 0.0307 | 4.962 | 0.8766 |
| nmos_off | v2 | False | targeted | 0.03056 | 0.3447 | 0.9954 |
| pmos_on | v1 | False | uniform | 0.02524 | 0.1629 | 0.9974 |
| pmos_on | v2 | True | uniform | 0.03517 | 0.142 | 0.9998 |
| pmos_on | v1 | False | targeted | 0.3543 | 654.1 | 0.8316 |
| pmos_on | v2 | True | targeted | 0.07072 | 0.773 | 0.9961 |
| pmos_off | v1 | True | uniform | 0.01165 | 0.07246 | 0.9974 |
| pmos_off | v2 | False | uniform | 0.01817 | 0.1037 | 0.9988 |
| pmos_off | v1 | True | targeted | 0.02209 | 4.48 | 0.9125 |
| pmos_off | v2 | False | targeted | 0.02636 | 0.364 | 0.9956 |

## Median relative error per terminal (v1 vs v2)

| case | test_set | terminal | v1 | v2 |
|---|---|---|---|---|
| nmos_off | targeted | ib | 0.02504 | 0.01968 |
| nmos_off | targeted | id | 0.04196 | 0.04164 |
| nmos_off | targeted | ig | 0.0129 | 0.01679 |
| nmos_off | targeted | is | 0.04292 | 0.04414 |
| nmos_off | uniform | ib | 0.00797 | 0.008732 |
| nmos_off | uniform | id | 0.02252 | 0.03057 |
| nmos_off | uniform | ig | 0.007001 | 0.01084 |
| nmos_off | uniform | is | 0.02228 | 0.03285 |
| nmos_on | targeted | ib | 0.02088 | 0.01932 |
| nmos_on | targeted | id | 0.5122 | 0.1262 |
| nmos_on | targeted | ig | 0.02576 | 0.02389 |
| nmos_on | targeted | is | 0.4655 | 0.1201 |
| nmos_on | uniform | ib | 0.01328 | 0.01703 |
| nmos_on | uniform | id | 0.03263 | 0.0541 |
| nmos_on | uniform | ig | 0.01133 | 0.01564 |
| nmos_on | uniform | is | 0.03382 | 0.05428 |
| pmos_off | targeted | ib | 0.01879 | 0.02705 |
| pmos_off | targeted | id | 0.02706 | 0.03284 |
| pmos_off | targeted | ig | 0.01135 | 0.01404 |
| pmos_off | targeted | is | 0.03117 | 0.03152 |
| pmos_off | uniform | ib | 0.008542 | 0.01226 |
| pmos_off | uniform | id | 0.01604 | 0.02282 |
| pmos_off | uniform | ig | 0.005524 | 0.01081 |
| pmos_off | uniform | is | 0.01648 | 0.02679 |
| pmos_on | targeted | ib | 0.0208 | 0.01984 |
| pmos_on | targeted | id | 0.6262 | 0.1163 |
| pmos_on | targeted | ig | 0.02395 | 0.02288 |
| pmos_on | targeted | is | 0.7462 | 0.1238 |
| pmos_on | uniform | ib | 0.01248 | 0.01696 |
| pmos_on | uniform | id | 0.04015 | 0.05516 |
| pmos_on | uniform | ig | 0.01005 | 0.0137 |
| pmos_on | uniform | is | 0.03829 | 0.05484 |

## |Vds| < 0.05 V (median relative error)

| case | model | test_set | n | id_med_rel | ig_med_rel | is_med_rel | ib_med_rel | id_p95_rel | is_p95_rel |
|---|---|---|---|---|---|---|---|---|---|
| nmos_on | v1 | uniform | 147 | 0.1271 | 0.02223 | 0.1273 | 0.02068 | 1.001 | 1.001 |
| nmos_on | v2 | uniform | 147 | 0.1158 | 0.02976 | 0.108 | 0.02365 | 0.5306 | 0.3997 |
| nmos_on | v1 | targeted | 615 | 1.001 | 0.03229 | 1.003 | 0.02549 | 9613 | 5539 |
| nmos_on | v2 | targeted | 615 | 0.3189 | 0.03348 | 0.2876 | 0.02461 | 3.086 | 9.213 |
| nmos_off | v1 | uniform | 167 | 0.02115 | 0.007434 | 0.02326 | 0.008618 | 0.5144 | 0.4235 |
| nmos_off | v2 | uniform | 167 | 0.0308 | 0.01294 | 0.03149 | 0.01011 | 0.2186 | 0.1708 |
| nmos_off | v1 | targeted | 629 | 0.03152 | 0.01274 | 0.03159 | 0.01352 | 12.38 | 146.7 |
| nmos_off | v2 | targeted | 629 | 0.03749 | 0.01887 | 0.04026 | 0.01223 | 0.5212 | 0.6606 |
| pmos_on | v1 | uniform | 146 | 0.2176 | 0.01746 | 0.2103 | 0.01889 | 6.331 | 3.45 |
| pmos_on | v2 | uniform | 146 | 0.09675 | 0.01741 | 0.1001 | 0.0232 | 0.864 | 0.6119 |
| pmos_on | v1 | targeted | 605 | 1.026 | 0.03218 | 1.038 | 0.02286 | 6757 | 3599 |
| pmos_on | v2 | targeted | 605 | 0.284 | 0.03493 | 0.3005 | 0.02541 | 2.56 | 2.369 |
| pmos_off | v1 | uniform | 141 | 0.01521 | 0.005839 | 0.01655 | 0.009399 | 0.4004 | 0.7176 |
| pmos_off | v2 | uniform | 141 | 0.02481 | 0.01106 | 0.02538 | 0.01417 | 0.1745 | 0.2315 |
| pmos_off | v1 | targeted | 619 | 0.02006 | 0.009942 | 0.0236 | 0.01328 | 5.11 | 46.03 |
| pmos_off | v2 | targeted | 619 | 0.02705 | 0.01383 | 0.02866 | 0.02103 | 0.417 | 0.593 |
