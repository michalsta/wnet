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
  Traces expand aggregate matches back to original peak capacities before
  recovering production dual potentials. The existing adjoint is not invariant
  to merged event blocks on full hemoglobin data: an initial grouped marginal
  reported 0 where a direct one-unit re-solve gives 1. Therefore the final
  marginal dispatcher preserves the original ungrouped event sequence. Grouping
  is enabled only for ordinary/certified solves, not for discrete marginals.
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
is shared baseline behavior, not evidence of non-Monge costs. No tested
ordinary matching objective has disagreed with its oracle. The separate
merged-adjoint discrepancy is reproducible with
`diagnose_grouped_marginals.cpp full-data.cst`; that diagnostic deliberately
bypasses the safe ungrouped marginal dispatcher. The `ExactPLF`/`ExactLinearPLF` diagnostic types and `test_window.cpp` check a
three-winner window implementation against direct enumeration. That isolated
correction does not resolve the merged-adjoint discrepancy, and is not enabled
in the benchmark dispatcher or production. Keep these findings distinct.

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
`--sample-seconds`, `--orders`, and `--repetitions` for isolated refinements or longer
follow-ups. Build a separate binary with `-DSWEEP_TRIAL_STATS` for untimed
work counters; timed builds omit counter updates. Keep fixtures, executables,
logs, plots and the generated comparison report outside the repository.

The final cursor refinement is selectable as `stream-cursor`,
`moment-cursor`, `group-linear-cursor` and `all-cursor` (the two moment modes
require p=2). It keeps a monotone evaluator over profit knots, reuses endpoint
values and uses a cached integer segment slope when exact. This preserves the
unfused subtraction's amortized access pattern instead of introducing repeated
binary searches during fusion. The same ungrouped marginal safeguard applies.

`test_full.cpp` optionally verifies an entire fixture with an independent LEMON
CostScaling model. It enumerates every profitable pair (default cap: 8 million;
exceeding the cap aborts rather than truncating). Build with the same include
paths as `test_weighted.cpp`, then run `test_full dataset.cst [pair_limit]`.
This is a correctness oracle, not a proposed production backend.

## Further NMR prototypes (2026-10-15)

The `nmr_*` files are a second isolated series based on production `a88135e`.
They do not change production dispatch or defaults. `nmr_kernel.hpp` is a
namespaced copy of that production kernel with curve policies and an adjoint
range entry point; `nmr_replay.hpp` adds replay without changing production.

- `flow`: LEMON CostScaling on every profitable pair of aggregated coordinates.
  Duals come directly from flow potentials. Exact discrete gains use one
  Dijkstra pass through the optimal residual graph; zero-supply theoretical
  locations and sufficient residual pair capacity are retained.
- `checkpoint64`, `checkpoint256`, `checkpoint1024`: save full states at that
  spacing and replay each block in reverse order for traceback or the adjoint.
  The final replay snapshot includes the next event's prune. Both orientations
  retain the production empirical-before-theoretical tie ordering.
- `virtual`: feed a read-only `V+T` knot-union view directly into the window
  operation and flip, eliminating its intermediate profile allocation.
- `groupdual`: recover on all unique grid coordinates and broadcast potentials
  to original peaks, retaining zero-supply coordinates for future weights.
- `quadratic`: use prefix mass/position/squared-position moments as a lazy T
  curve, then feed the virtual sum into window/flip. Intermediate moments use
  128 or 256 bits as needed. The final DP envelopes remain materialized.
- `checkpoint256-groupdual`, `virtual-checkpoint256`,
  `quadratic-checkpoint256`: combinations. The latter two use grouped duals.

Build `nmr_test.cpp` (asserts enabled) and `nmr_bench.cpp` (-O3 -DNDEBUG) with
`-std=c++20 -Isrc/wnet/cpp -Isrc/wnet/cpp/wnet -I<installed-pylmcf>/cpp`.
The test uses its own all-pairs capacitated-flow oracle, explicit +1 re-solves,
all-pair dual checks, virtual-curve enumeration, wide moments, split events,
empty sides, and small plus >1024-event replay boundaries.

`nmr_bench fixture.cst variant solve|dual|gains|both [reps=3] [pair_limit=8000000]
[warmup=1]` prints readiness, per-sample values and a final median. `both`
measures a complete cost-and-gain evaluation; flow reuses its one solved flow,
whereas the sweep runs its ordinary solve and two marginal orientations.
A flow `gains` call includes constructing/solving its primal (conservative
relative to sweep gains, which excludes the preceding ordinary solve).
`dual-compare` on a checkpoint variant reuses one traceback to measure original
and grouped dual recovery separately. Its overall time includes BOTH recovery
calls and is not an ordinary dual timing.

`nmr_run.py --work <dir> --fixtures <dir> --stage small|ordinary|heavy` runs
sequentially on CPU 4. This series sets **no address-space cap**. It samples
VmRSS/VmHWM and VmSize every 20 ms, terminates above an 8 GiB sampled resident
budget, and records censored outcomes. Very short processes may have
under-sampled memory peaks. Explicit flow graph construction has an 8-million
pair budget per component and rejects rather than truncates. Small fixtures
use one warmup and three measured calls; heavy full-data attempts use one
measured call without warmup. Timers include grouping, graph/moment setup and
per-call work; fixture loading and cost-model construction are excluded.
