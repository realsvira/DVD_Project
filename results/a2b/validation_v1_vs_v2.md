# Part A2b: v1 vs v2 on validation

Scores = mean over Id/Ig/Is/Ib of median relative error. `val_orig` = original 15 % validation split (spec distribution); `val_targeted` = 20 % of the targeted supplementary samples; `val_smallvds_*` = |Vds| < 0.05 V rows of both.

| case | model | score | val_orig | val_targeted | val_smallvds_id_med_rel | val_smallvds_is_med_rel |
|---|---|---|---|---|---|---|
| nmos_on | v1 (A2) | 0.1529 | 0.02255 | 0.2832 | 1 | 1 |
| nmos_on | v2 #0 | 0.05319 | 0.03485 | 0.07153 | 0.2448 | 0.2144 |
| nmos_on | v2 #1 | 0.05258 | 0.03505 | 0.07011 | 0.2331 | 0.2127 |
| nmos_on | v2 #2 | 0.06865 | 0.04425 | 0.09305 | 0.2967 | 0.2903 |
| nmos_off | v1 (A2) | 0.02216 | 0.01546 | 0.02886 | 0.02438 | 0.02925 |
| nmos_off | v2 #0 | 0.02529 | 0.0223 | 0.02828 | 0.03046 | 0.03278 |
| nmos_off | v2 #1 | 0.02429 | 0.02028 | 0.0283 | 0.0355 | 0.0317 |
| nmos_off | v2 #2 | 0.02502 | 0.02045 | 0.02959 | 0.03183 | 0.03667 |
| pmos_on | v1 (A2) | 0.2327 | 0.02569 | 0.4396 | 1 | 1 |
| pmos_on | v2 #0 | 0.059 | 0.03629 | 0.08172 | 0.2467 | 0.2537 |
| pmos_on | v2 #1 | 0.05576 | 0.03578 | 0.07575 | 0.2303 | 0.2089 |
| pmos_on | v2 #2 | 0.06344 | 0.03857 | 0.08831 | 0.2587 | 0.2728 |
| pmos_off | v1 (A2) | 0.0163 | 0.0115 | 0.02109 | 0.01648 | 0.02142 |
| pmos_off | v2 #0 | 0.02258 | 0.01814 | 0.02702 | 0.02646 | 0.03199 |
| pmos_off | v2 #1 | 0.02176 | 0.01783 | 0.0257 | 0.02389 | 0.03013 |
| pmos_off | v2 #2 | 0.02093 | 0.01719 | 0.02467 | 0.02308 | 0.02374 |

| case | selected | v1_score | v2_score | v2_config | file_sha256 |
|---|---|---|---|---|---|
| nmos_on | v2 | 0.1529 | 0.05258 | v2 #1 | 7f7a14c325bc2e19d3f0f8da17d9f32ea07febea2efe270e58accdebd39c2510 |
| nmos_off | v1 | 0.02216 | 0.02429 | v2 #1 | 3bf63f03c70a5b178f2eaec67140a12defd2040e926fb2e9e5a420eb7fc408b6 |
| pmos_on | v2 | 0.2327 | 0.05576 | v2 #1 | ec74543f32f692233577b960f6a8b11b473f89646720192ab28c0834a1388d5f |
| pmos_off | v1 | 0.0163 | 0.02093 | v2 #2 | 25036a35baade4f90b4af15180b3823ded61579407c170beffd9ba10154a7d8c |
