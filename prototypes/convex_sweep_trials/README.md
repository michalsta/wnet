# ConvexSweep optimization trials

These are isolated prototypes based on production commit `e645fad`. They do
not change production dispatch, Python APIs, defaults or installed extensions.
All costs come from the production Monge-preserving `GridPower` model. No
non-Monge matching repair or alternative rounded-pair model is used.

`kernel.hpp` is a namespaced experimental copy of `convex_sweep.hpp`, with
compile-time profile and transition policies. Keeping the copy here permits
side-by-side tests and benchmarks without making production carry experimental
branches. A selected change should subsequently be ported into production and
checked there; these files are not a drop-in patch to merge wholesale.

## Variants

- `baseline`: the production flat-profile operations and recurrence, in the
  experimental namespace. Tests compare it with the production kernel too.
- `group`: aggregate supplies with identical grid coordinate and side, after
  integer quantization. Do not cancel empirical against theoretical supply.
  Grouping includes zero-count theoretical events for marginal probes. Traces
  expand aggregate matches back to original peak capacities before recovering
  production dual potentials.
- `linear`: merge sorted breakpoint streams, use segment cursors for maxima,
  normalize in place, reuse pointwise-max buffers, and shift/truncate arrays in
  place. Generic integer crossing searches are retained.
- `fused`: emit `window_max(V+T)-T` in one pass and move flip profiles rather than
  copying them. This first version includes T knots in the window candidates;
  it can increase window work despite eliminating an intermediate profile.
- `moments`: quadratic-only lazy profit curve, using supply-mass and first/
  second position moments. It requires the fused interface. Prefix moments
  use centered coordinates and 128-bit arithmetic, switching to 256 bits when
  necessary to protect cancellation terms. The final integer result is checked.
- `stream`: a refined fusion. Stream window-profile vertices into the
  subtraction merge; interpolate at T knots without reevaluating the window.
- `moment-stream`: the refined fusion with cached end moments and direct
  evaluation of moment knots. Other moment queries still use prefix searches.

`group-linear`, `group-fused`, `linear-fused`, `group-linear-fused`,
`group-stream`, `linear-stream`, `group-linear-stream`, `all` and `all-stream`
select corresponding combinations. `all` and `all-stream` include moments and
therefore require p=2. For other orders, the group/linear/fusion combinations
remain applicable.

The quadratic prototype eliminates the explicit T construction; it does not
eliminate materialization of V+T or the general DP envelope. No O(n log n)
whole-solver claim follows from the moment data structure. In marginal runs,
the new transition policies affect the forward pass; the adjoint retains the
existing recurrence. Flat-profile improvements also affect the adjoint.

## Build and verify

From the repository root, with Boost headers and the installed pylmcf headers:

```sh
c++ -std=c++20 -O3 -DNDEBUG -Isrc/wnet/cpp -Isrc/wnet/cpp/wnet \
    prototypes/convex_sweep_trials/bench.cpp -o /tmp/convex-trial-bench
c++ -std=c++20 -O2 -Isrc/wnet/cpp -Isrc/wnet/cpp/wnet \
    prototypes/convex_sweep_trials/test.cpp -o /tmp/convex-trial-test
/tmp/convex-trial-test 4000
```

`test_weighted.cpp` additionally needs `-I<installed-pylmcf>/cpp` and checks
large capacities against an independent LEMON cost-scaling flow oracle. Its
inputs use the same structurally Monge model, including fractional powers.
Tests cover every applicable variant, unrestricted small matching oracles,
weighted flow oracles, traced profits, capacity-valid expanded matches,
primal-dual equality, all pair inequalities, zero-weight target activation,
exact grid coincidences, one-unit marginal probes and wide moment arithmetic.
Compile `test.cpp` with `-DWNET_FORCE_SOFT_INT128 -fsanitize=undefined` to check
the software-wide production arithmetic too. Do not define NDEBUG for tests.

`diagnose_window.cpp` is a separate diagnostic: the existing production generic
window operation can overestimate the maximum of an arbitrary PL profile when
an edge crosses an interior plateau. It prints a small counterexample. This
is shared baseline behavior, not evidence of non-Monge costs; none of the
reachable-state oracle suites has reproduced an objective discrepancy. This
experiment does not repair production's generic window implementation. Keep
that limitation distinct from the optimization test results.

## Full-data benchmarking

`export.cpp` reads normalized full spectra in the existing frozen raw format:
uint64 target count, double radius/intensity scale, then each spectrum as a
uint64 count followed by double positions and normalized intensities. Build
against the installed pylmcf headers. It invokes the current production 1D
factory, traverses each native chain in its native orientation and preserves
all component geometry, quantized capacities and theoretical-probe locations.
It writes a versioned `.cst` fixture. A positive scale argument fixes S; zero
uses the current automatic scale. Each side's trash price is radius**p/2;
weights are uniform and intensity scale is 1e8, matching the prior full-data
benchmark settings.

```sh
/tmp/convex-trial-export full-spectrum.raw dataset.cst 2 460508
/tmp/convex-trial-bench dataset.cst group-linear-stream solve 3
/tmp/convex-trial-bench dataset.cst group-linear-stream dual 3
/tmp/convex-trial-bench dataset.cst group-linear-stream gains 3
```

Native times exclude loading, graph construction and geometry/model setup.
They include event construction/grouping, scratch-buffer work, moment-index
construction when selected, and trace expansion plus dual recovery in `dual`.
`gains` measures both orientations of forward/adjoint integer marginal gains,
excluding the preceding ordinary solve. Repeated results are checked exactly.
These timings must not be compared directly with full Python-network timings.

`run.py` runs the five full frozen datasets sequentially on nighthaven CPU 4.
The default initial budget is 15 seconds **per sample**, with a separate
30-second fixture-setup budget and a 4 GiB address-space cap. It uses one warmup
and three measured calls, resumes completed cases, stores exact result values,
raw JSONL output, errors and process peak RSS, and skips later phases when an
ordinary measurement does not finish. Timeout statuses are not timings.
The same fixture and scale are shared by every variant. The additional
hemoglobin p=2 auto-scale fixture checks high precision separately.

Use `--variants`, `--bench`, `--results-name`, `--phases`, `--datasets`,
`--sample-seconds`, and `--repetitions` for isolated refinements or longer
follow-ups. Build a separate binary with `-DSWEEP_TRIAL_STATS` for untimed
work counters; timed builds omit counter updates. Keep fixtures, executables,
logs, plots and the generated comparison report outside the repository.
