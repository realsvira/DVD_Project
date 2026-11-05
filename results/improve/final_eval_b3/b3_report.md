# Part B step 3: gate-level SPICE references

Generated 2026-11-13T14:25:50, ngspice 44.2, model `45nm_HP.pm`. 3600 simulations (18 gate/input cases x 200 corners) in 355 s.
These results are evaluation references only; no model is trained on them.

## Checks

- PASS: 3600 simulations == 3600 expected
- PASS: converged: 3600/3600
- PASS: truth table correct: 3600/3600
- PASS: power balance sources == devices (max rel err 7.8e-07 < 1e-06)
- PASS: leakage positive and finite
- PASS: unique (gate, inputs, corner)

## Leakage definition

Each source V_k (VDD rail and each input) delivers I_k = −i(V_k). P_leak = Σ V_k·I_k and **I_leak = P_leak / VDD** (positive). It includes sub-threshold, gate-tunnelling (also from inputs held at VDD) and junction leakage. The VDD-rail current alone (`i_vdd`) and the input currents (`i_in_a`, `i_in_b`) are stored too. Every device terminal has a 0 V probe; Σ over devices of V·I equals P_leak (power-balance check above).

## Sizing and topology (W not specified by the PDF)

Equal-drive sizing: unit inverter Wn = 2·Lmin = 90 nm, Wp = 4·Lmin = 180 nm; series stacks of two use 2x unit width; parallel devices unit width. NAND2: MNA (gate A) on top of MNB (gate B); NOR2: MPA (gate A) at VDD above MPB (gate B). AND2/OR2 = NAND2/NOR2 + inverter, simulated as one complete netlist.

| gate | device | type | W (nm) | d | g | s | b |
|---|---|---|---|---|---|---|---|
| INV | mp | pmos | 180 | out | a | vdd | vdd |
| INV | mn | nmos | 90 | out | a | 0 | 0 |
| NAND2 | mpa | pmos | 180 | out | a | vdd | vdd |
| NAND2 | mpb | pmos | 180 | out | b | vdd | vdd |
| NAND2 | mna | nmos | 180 | out | a | x | 0 |
| NAND2 | mnb | nmos | 180 | x | b | 0 | 0 |
| NOR2 | mpa | pmos | 360 | y | a | vdd | vdd |
| NOR2 | mpb | pmos | 360 | out | b | y | vdd |
| NOR2 | mna | nmos | 90 | out | a | 0 | 0 |
| NOR2 | mnb | nmos | 90 | out | b | 0 | 0 |
| AND2 | mpa | pmos | 180 | n1 | a | vdd | vdd |
| AND2 | mpb | pmos | 180 | n1 | b | vdd | vdd |
| AND2 | mna | nmos | 180 | n1 | a | x | 0 |
| AND2 | mnb | nmos | 180 | x | b | 0 | 0 |
| AND2 | mpi | pmos | 180 | out | n1 | vdd | vdd |
| AND2 | mni | nmos | 90 | out | n1 | 0 | 0 |
| OR2 | mpa | pmos | 360 | y | a | vdd | vdd |
| OR2 | mpb | pmos | 360 | n1 | b | y | vdd |
| OR2 | mna | nmos | 90 | n1 | a | 0 | 0 |
| OR2 | mnb | nmos | 90 | n1 | b | 0 | 0 |
| OR2 | mpi | pmos | 180 | out | n1 | vdd | vdd |
| OR2 | mni | nmos | 90 | out | n1 | 0 | 0 |

## Truth table (all corners)

| gate | inputs | n | logic_ok | v_out_min | v_out_max |
|---|---|---|---|---|---|
| AND2 | 00 | 200 | 200 | -4.132e-07 | 9.148e-05 |
| AND2 | 01 | 200 | 200 | 1.249e-07 | 9.163e-05 |
| AND2 | 10 | 200 | 200 | 1.249e-07 | 9.155e-05 |
| AND2 | 11 | 200 | 200 | 0.9033 | 1.1 |
| INV | 0 | 200 | 200 | 0.9033 | 1.1 |
| INV | 1 | 200 | 200 | 1.249e-07 | 9.147e-05 |
| NAND2 | 00 | 200 | 200 | 0.9033 | 1.1 |
| NAND2 | 01 | 200 | 200 | 0.9032 | 1.1 |
| NAND2 | 10 | 200 | 200 | 0.9033 | 1.1 |
| NAND2 | 11 | 200 | 200 | 2.686e-07 | 0.0001767 |
| NOR2 | 00 | 200 | 200 | 0.9032 | 1.1 |
| NOR2 | 01 | 200 | 200 | 2.373e-07 | 0.0001878 |
| NOR2 | 10 | 200 | 200 | 8.288e-08 | 6.679e-05 |
| NOR2 | 11 | 200 | 200 | 6.975e-08 | 4.859e-06 |
| OR2 | 00 | 200 | 200 | 1.249e-07 | 9.163e-05 |
| OR2 | 01 | 200 | 200 | 0.9033 | 1.1 |
| OR2 | 10 | 200 | 200 | 0.9033 | 1.1 |
| OR2 | 11 | 200 | 200 | 0.9033 | 1.1 |

## Leakage per gate and input (A, over 200 corners)

| gate | inputs | median | mean | min | p5 | p95 | max |
|---|---|---|---|---|---|---|---|
| AND2 | 00 | 3.023e-09 | 4.376e-09 | 2.091e-10 | 5.383e-10 | 1.239e-08 | 2.566e-08 |
| AND2 | 01 | 7.874e-09 | 1.251e-08 | 4.036e-10 | 1.343e-09 | 4.06e-08 | 7.401e-08 |
| AND2 | 10 | 4.494e-09 | 7.487e-09 | 2.44e-10 | 8.66e-10 | 2.548e-08 | 4.758e-08 |
| AND2 | 11 | 6.899e-09 | 1.061e-08 | 3.559e-10 | 1.331e-09 | 3.285e-08 | 6.754e-08 |
| INV | 0 | 2.735e-09 | 4.731e-09 | 1.917e-10 | 5.479e-10 | 1.424e-08 | 2.486e-08 |
| INV | 1 | 1.524e-09 | 2.845e-09 | 7.326e-11 | 2.069e-10 | 1.057e-08 | 2.129e-08 |
| NAND2 | 00 | 1.162e-09 | 1.48e-09 | 1.321e-10 | 2.48e-10 | 4.212e-09 | 5.484e-09 |
| NAND2 | 01 | 5.583e-09 | 9.66e-09 | 3.303e-10 | 9.06e-10 | 3.019e-08 | 5.334e-08 |
| NAND2 | 10 | 2.592e-09 | 4.64e-09 | 1.708e-10 | 5.26e-10 | 1.527e-08 | 2.694e-08 |
| NAND2 | 11 | 3.208e-09 | 5.871e-09 | 1.642e-10 | 4.632e-10 | 2.118e-08 | 4.281e-08 |
| NOR2 | 00 | 7.033e-09 | 1.051e-08 | 4.99e-10 | 1.327e-09 | 2.994e-08 | 5.103e-08 |
| NOR2 | 01 | 4.175e-09 | 6.787e-09 | 2.271e-10 | 6.326e-10 | 2.222e-08 | 4.725e-08 |
| NOR2 | 10 | 1.05e-09 | 2.039e-09 | 4.964e-11 | 1.409e-10 | 7.708e-09 | 1.557e-08 |
| NOR2 | 11 | 5.722e-10 | 7.308e-10 | 6.222e-11 | 1.301e-10 | 1.875e-09 | 2.521e-09 |
| OR2 | 00 | 9.496e-09 | 1.336e-08 | 6.338e-10 | 1.733e-09 | 4.184e-08 | 7.17e-08 |
| OR2 | 01 | 8.18e-09 | 1.152e-08 | 5.064e-10 | 1.564e-09 | 3.485e-08 | 6.784e-08 |
| OR2 | 10 | 4.299e-09 | 6.771e-09 | 2.414e-10 | 7.842e-10 | 2.226e-08 | 3.841e-08 |
| OR2 | 11 | 3.601e-09 | 5.462e-09 | 3.1e-10 | 7.493e-10 | 1.581e-08 | 2.723e-08 |
