# Part A1 smoke test

ngspice 44.2, model `45nm_HP.pm`, L = 45 nm. Currents in A, positive = into the device terminal.

## Nominal (VDD = 1.0 V, T = 27 C, W = 90 nm, nominal process)

```
    case  vdd  temp            w  vd  vs  vg  vb            id            ig            is            ib status  standalone_max_rel_diff       kcl_sum      kcl_rel
 nmos_on  1.0  27.0 9.000000e-08 1.0 0.0 1.0 0.0  1.120762e-04  5.887824e-11 -1.120762e-04 -3.864824e-11     ok             6.963038e-13 -2.445578e-17 2.182067e-13
nmos_off  1.0  27.0 9.000000e-08 1.0 0.0 0.0 0.0  1.694675e-09 -1.280122e-11 -1.681863e-09 -1.018644e-14     ok             1.623318e-12  4.763347e-25 2.810774e-16
 pmos_on  1.0  27.0 9.000000e-08 0.0 1.0 0.0 1.0 -7.942990e-05 -1.172555e-10  7.943000e-05  2.113965e-11     ok             7.029527e-13 -3.492486e-17 4.396936e-13
pmos_off  1.0  27.0 9.000000e-08 0.0 1.0 1.0 1.0 -4.212503e-10  5.871836e-11  3.625218e-10  1.004752e-14     ok             4.213008e-12 -6.994058e-17 1.660309e-07
```

## Corner sweep

```
                 n    failed   min|Id|   max|Id|   max|Ig|   max|Ib|  max|KCL sum|
case                                                                              
nmos_on  4.800e+01 0.000e+00 6.466e-16 7.227e-04 1.269e-08 1.156e-08     1.177e-15
nmos_off 4.800e+01 0.000e+00 6.013e-14 2.116e-07 5.579e-10 9.380e-12     6.293e-23
pmos_on  4.800e+01 0.000e+00 7.658e-14 6.603e-04 1.126e-08 7.198e-09     1.365e-16
pmos_off 4.800e+01 0.000e+00 3.668e-13 7.792e-08 2.026e-09 8.939e-12     1.418e-15
```

## Checks

- PASS: all 4 nominal simulations converged
- PASS: all currents finite
- PASS: Kirchhoff: |Id+Ig+Is+Ib| <= 1e-06*max|I| + 1e-14 A
- PASS: batch engine == stand-alone example netlist (rel diff < 1e-9)
- PASS: sign of Id: NMOS current enters drain, PMOS current leaves drain
- PASS: NMOS Ion/Ioff = 6.61e+04 (> 1e3)
- PASS: PMOS Ion/Ioff = 1.89e+05 (> 1e3)
- PASS: corner sweep: 192/192 converged
- PASS: corner sweep: Kirchhoff holds at every corner

**PASSED**
