# Improvement phase: pre-declared protocol (written 2026-11-13T13:40:59+05:30, before any experiment ran)

## Data roles
- Training: original 70 % train split + 80 % of targeted supplementary samples (as v2). Optional extra
  transistor-level samples only if the learning-curve experiment shows a benefit.
- Validation (all decisions): original 15 % val split ("val_orig") + 20 % of targeted samples ("val_targeted").
- NOT used for any decision or training: original test split, A2b fresh test sets, Part B gate references,
  Part C circuit references (all already seen), and the NEW final-evaluation data in data/final_eval/
  (fresh transistor tests salt 4/5, 200 new corners seed 5005 with gate + CLA SPICE references), which is
  evaluated exactly once at the end.

## Selection score (per transistor case, lower is better)
For each validation set v in {val_orig, val_targeted}:
  s_v = mean over {Id, Ig, Is, Ib} of median relative error + 0.1 * mean over {Id, Ig, Is, Ib} of p95 relative error
S = (s_val_orig + s_val_targeted) / 2
(relative error = |I_pred - I_spice| / max(|I_spice|, 1 fA), as in Part A2)
The candidate with the lowest S is selected per case; the current final model (v2) is always a candidate.

## Final reporting
Before (current final models) and after (selected models) on: (1) the new final-evaluation data (primary,
untouched), and (2) the previously seen sets (secondary, labelled as already seen).
