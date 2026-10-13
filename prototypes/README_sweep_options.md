# Sweep quantization experiments

Historical experiment (12 October 2026): these alternatives compare against
independently rounded pair costs. Production ConvexSweep now uses
[Monge-preserving quantization](../docs/monge_quantization.md). Its withdrawn
repair is retained in `non_monge_repair.hpp` solely to reproduce this baseline.
Use the frozen exported inputs for the published comparison; new exports use
the currently installed production factory and its current scales.


These standalone programs compare alternatives to the production rounding
correction. They do not change production solver dispatch. They currently
support p=1 and p=2 and independent symmetric trash.

Build against the production headers and the installed pylmcf headers:

```sh
g++ -std=c++20 -O3 -DNDEBUG -Isrc/wnet/cpp \
    -I"$(python -m pylmcf -i)" prototypes/bench_sweep_options.cpp -o /tmp/bench_options
g++ -std=c++20 -O1 -g -DWNET_FORCE_SOFT_INT128 \
    -fsanitize=undefined -fno-sanitize-recover=undefined -Isrc/wnet/cpp \
    -I"$(python -m pylmcf -i)" prototypes/test_sweep_options.cpp -o /tmp/test_options
/tmp/test_options
```

`export_sweep_options.py input.npz output.raw` reads full arrays from the
previous real-data benchmark format and its sibling `manifest.json`. It uses
unit-mass spectra, uniform mixture weights and intensity scale 1e8. The native
export builds the production 1D graph to retain its exact components, original
chain orientation/positions, cost scale, integer capacities and isolated trash:

```sh
python prototypes/export_sweep_options.py /path/hemoglobin.npz /tmp/hemoglobin.raw
/tmp/bench_options export /tmp/hemoglobin.raw /tmp/hemoglobin-p1.bin 1
/tmp/bench_options export /tmp/hemoglobin.raw /tmp/hemoglobin-p2.bin 2
```

Native arguments are `components.bin mode repetitions [grid_step]`. One warmup
precedes the timed repetitions. Modes:

- `baseline`: production native cycle correction, using specialized p=1/p=2
  pair arithmetic rather than a generic `pow` call.
- `cached`: the same correction with cached profitable pairs and inactive
  vertices removed from residual searches.
- `cost-scaling`: an exact explicit profitable-pair circulation with LEMON
  cost scaling and its matching dual.
- `active`: an exact restricted circulation; initialize nearest targets, solve,
  price all omitted profitable pairs, and add each row's worst violation until
  the full positive-capacity problem is certified.
- `approx`: keep the sweep matching, relax complementary equalities by epsilon,
  but enforce original pair inequalities exactly. Report an explicit dual gap.
- `grid`: round coordinates, then use integer linear/squared distance and
  consistently rounded matching reward. At p=1 use SlopeDP; at p=2 use the
  sweep. Project the certificate to the original pair-cost model.
- `slope`: p=1 chain-gap quantization with a certificate projected to the
  original pair-cost model.
- `grid-model`: the grid model with its own exact certificate, without
  projection to the original costs.
- `grid-sweep-model`: the same grid model, using ConvexSweep also at p=1.
- `slope-native`: the chain model with its own exact certificate.
- `grid-fast` and `slope-fast`: p=1 original-cost certificates using linear
  prefix/suffix maxima, a half-unit rounding allowance and a conservative
  floating-point guard, instead of enumerating all pairs for projection.

For the last three modes, `model_cost`/`model_gap` describe the changed model;
`upper` evaluates their feasible matching using original costs. `lower` and
`gap` are null because no certificate for the original model was requested.
Changed-model trash coefficients are quantized with its matching reward.
Isolated trash retains its original price. The grid step can increase at p=2
to protect the integer objective; the effective maximum step is reported.

The benchmark includes scratch/pair construction and certificate work, and
excludes graph export, input loading and the Python wrapper. Repeated runs do
not reuse a simplex basis or candidate cache across solves. The exact explicit
and cached alternatives cap profitable pairs at eight million. Active pricing
caps rounds at 128. These are experimental resource limits.

`run_sweep_options.py executable component_directory output_directory` runs the
full matrix sequentially on CPU 4, with a 4 GiB address limit and 30 seconds
per warmup/timed sample. `--models-only` runs the changed-model certificate
variants. Results, raw samples, errors and RSS are written outside the repo;
rerunning resumes completed cases. Do not interpret a timeout as a timing.

Tests use exhaustive weighted matching on small randomized inputs, validate
both exact and bounded certificates against all pair inequalities and future
one-unit target probes, check discrete derivatives of the exact alternatives,
check the grid models against their own oracle, and exercise candidate growth.
The bounds refer to the fixed integer supplies. A production integration also
needs the existing continuous-supply rounding budget and Python API contract.
