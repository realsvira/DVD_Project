# Part B step 3: gate-level SPICE references

Generated 2026-11-13T10:53:20, ngspice 44.2, model `45nm_HP.pm`. 3600 simulations (18 gate/input cases x 200 corners) in 711 s.
These results are evaluation references only; no model is trained on them.

## Checks

- PASS: 3600 simulations == 3600 expected
- PASS: converged: 3600/3600
- PASS: truth table correct: 3600/3600
- PASS: power balance sources == devices (max rel err 9.0e-07 < 1e-06)
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
| AND2 | 00 | 200 | 200 | 1.758e-07 | 0.0001004 |
| AND2 | 01 | 200 | 200 | 1.758e-07 | 0.0001007 |
| AND2 | 10 | 200 | 200 | 1.758e-07 | 0.0001005 |
| AND2 | 11 | 200 | 200 | 0.9006 | 1.1 |
| INV | 0 | 200 | 200 | 0.9006 | 1.1 |
| INV | 1 | 200 | 200 | 1.758e-07 | 0.0001004 |
| NAND2 | 00 | 200 | 200 | 0.9006 | 1.1 |
| NAND2 | 01 | 200 | 200 | 0.9006 | 1.1 |
| NAND2 | 10 | 200 | 200 | 0.9006 | 1.1 |
| NAND2 | 11 | 200 | 200 | 3.884e-07 | 0.0001935 |
| NOR2 | 00 | 200 | 200 | 0.9006 | 1.1 |
| NOR2 | 01 | 200 | 200 | 3.413e-07 | 0.000206 |
| NOR2 | 10 | 200 | 200 | 1.101e-07 | 7.442e-05 |
| NOR2 | 11 | 200 | 200 | 6.144e-08 | 4.696e-06 |
| OR2 | 00 | 200 | 200 | 1.758e-07 | 0.0001007 |
| OR2 | 01 | 200 | 200 | 0.9006 | 1.1 |
| OR2 | 10 | 200 | 200 | 0.9006 | 1.1 |
| OR2 | 11 | 200 | 200 | 0.9006 | 1.1 |

## Leakage per gate and input (A, over 200 corners)

| gate | inputs | median | mean | min | p5 | p95 | max |
|---|---|---|---|---|---|---|---|
| AND2 | 00 | 2.627e-09 | 4.113e-09 | 2.255e-10 | 4.795e-10 | 1.223e-08 | 2.568e-08 |
| AND2 | 01 | 7.476e-09 | 1.254e-08 | 5.699e-10 | 1.077e-09 | 4.058e-08 | 8.408e-08 |
| AND2 | 10 | 4.569e-09 | 7.443e-09 | 3.036e-10 | 6.388e-10 | 2.311e-08 | 4.712e-08 |
| AND2 | 11 | 6.618e-09 | 1.042e-08 | 4.749e-10 | 9.744e-10 | 3.152e-08 | 6.34e-08 |
| INV | 0 | 2.564e-09 | 4.765e-09 | 1.787e-10 | 3.927e-10 | 1.668e-08 | 3.317e-08 |
| INV | 1 | 1.418e-09 | 2.743e-09 | 9.561e-11 | 2.186e-10 | 9.673e-09 | 2.148e-08 |
| NAND2 | 00 | 1.05e-09 | 1.369e-09 | 1.067e-10 | 2.051e-10 | 3.76e-09 | 6.223e-09 |
| NAND2 | 01 | 5.126e-09 | 9.788e-09 | 3.226e-10 | 7.13e-10 | 3.443e-08 | 6.94e-08 |
| NAND2 | 10 | 2.352e-09 | 4.698e-09 | 1.642e-10 | 3.621e-10 | 1.701e-08 | 3.249e-08 |
| NAND2 | 11 | 3.046e-09 | 5.647e-09 | 2.237e-10 | 4.794e-10 | 1.947e-08 | 4.322e-08 |
| NOR2 | 00 | 6.452e-09 | 1.043e-08 | 4.784e-10 | 1.116e-09 | 3.478e-08 | 6.957e-08 |
| NOR2 | 01 | 3.572e-09 | 6.446e-09 | 3.119e-10 | 5.679e-10 | 2.047e-08 | 4.65e-08 |
| NOR2 | 10 | 9.973e-10 | 1.967e-09 | 6.419e-11 | 1.438e-10 | 6.876e-09 | 1.595e-08 |
| NOR2 | 11 | 5.151e-10 | 6.553e-10 | 6.513e-11 | 1.301e-10 | 1.592e-09 | 2.596e-09 |
| OR2 | 00 | 8.622e-09 | 1.318e-08 | 7.118e-10 | 1.372e-09 | 4.024e-08 | 8.424e-08 |
| OR2 | 01 | 7.574e-09 | 1.121e-08 | 5.497e-10 | 1.236e-09 | 3.328e-08 | 6.593e-08 |
| OR2 | 10 | 4.28e-09 | 6.733e-09 | 3.02e-10 | 5.748e-10 | 2.1e-08 | 4.336e-08 |
| OR2 | 11 | 3.382e-09 | 5.421e-09 | 2.602e-10 | 6.034e-10 | 1.753e-08 | 3.57e-08 |
