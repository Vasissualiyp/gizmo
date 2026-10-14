# rootfind_unit

Unit test for `system/bracketed_rootfind.h`, the Brent root-finder block that `DoCooling` (gas
temperature), `rt_ir_lambdadust` and `rt_eqm_dust_temp` (dust temperature) include.

`test_rootfind_unit.cc` stubs the few GIZMO symbols the header uses and runs it on twelve bracketed
problems shaped like the residuals those callers produce: a residual spanning nine decades across the
bracket, |f| ≈ 0 at one end for reasons unrelated to the root, a cubic, a bracket on a negative variable
(x = Tdust − T), jumps (opacity-table zone edges), a flat cubic root, a root at an end point, a very steep
root, and two non-bracketing intervals that exercise the expansion fallback for positive and negative
variables. Each case must return a point the block actually evaluated, within 3× the requested tolerance
of the known root, under the iteration cap, without `endrun()`.

`test_rootfind_unit.py` compiles and runs it twice, with `-O2` and with the production flags
(`-O3 -ffast-math -funroll-loops`). No build of GIZMO is required.

Written 2026-10-13 after the dust-temperature stress test found that the block returned an unevaluated
interpolation step when the proposed step was shorter than the tolerance (returning an end point for a
root 6× away) and that its expansion fallback walks away from the root for negative variables.
