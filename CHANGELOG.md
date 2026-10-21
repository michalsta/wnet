# Release notes

## Unreleased

### New APIs

- `StartMode` and `NetworkSimplex.start`: how a cold solve begins.
  `StartMode.Trash` (the new default) crash-starts every subgraph from the
  flow that sends all mass to the trash, with the `Source->Emp` and
  `Theo->Sink` anchors as the starting tree, so every matching arc is priced
  at its distance minus the trash cost it saves. `StartMode.Artificial` is
  LEMON's artificial basis, as before. Answers are identical. It applies to
  the first solve and to every warm-restart cold fallback, for all trash
  models; with no trash model, or with `WarmMode.LinkCut`, the artificial
  start is used. `crash_start_count()` on the network counts these starts.
  Needs a pylmcf with `crashRun` (reported as `wnet_cpp.HAS_CRASH_START`);
  against older pylmcf `StartMode.Trash` falls back to the artificial start.
- `NetworkSimplex(...)` takes every field as a keyword argument (pivot, warm,
  start, warm_violation_limit, repair_max_nodes, warm_violation_mass_fraction,
  warm_repair_budget, large_repair_mass_fraction), with the defaults of a
  plain `NetworkSimplex()`.
- `NetworkSimplex`, `NSPivotRule`, `WarmMode`, `StartMode` and the other
  solver configs (`CostScaling`, `CycleCanceling`, `CapacityScaling`,
  `SlopeDP`, `ConvexSweep`) are importable from `wnet` directly.
- `WassersteinNetwork` forwards `warm_start_count()`, `cold_start_count()`,
  `crash_start_count()`, `dual_repair_count()` and `primal_repair_count()`.
- `dual_cut()` reports allowances for cost quantization alongside its
  quantized-cost certificate (whose values are unchanged):
  `cost_rounding_error` (how far the current solution's real cost can exceed
  its quantized cost), `cost_round_up_ratio` and `cost_lower_slack`. Together
  they turn the cut into one for the real-cost problem; see
  [Dual cuts](docs/dual_cuts.md#cost-quantization). A coarse cost grid now shows
  up as a large `cost_rounding_error` instead of passing unnoticed.
- Violated-mass warm/cold gates on `NetworkSimplex` (need a pylmcf with
  `setWarmViolationMassFraction`, reported as `wnet_cpp.HAS_WARM_MASS_GATE`;
  inert otherwise): `warm_violation_mass_fraction` sends every subgraph cold
  when the violated basic-arc mass left by the tree patch exceeds that
  fraction of its positive supply, and `large_repair_mass_fraction` applies
  the same gate only to subgraphs above `repair_max_nodes` (which otherwise
  never repair). `warm_repair_budget` forwards the in-flight repair budget.
  All default to -1 (unchanged behaviour). `set_warm_policy(config)` changes
  these fields between solves without a rebuild (refined copies inherit
  them), and `ns_stats()` reports work units, policy cold starts and the
  last violation measurements.
- `dual_cut()` reports `transfer_round_up_ratio`, `transfer_round_down_ratio`
  and `transfer_absolute_error`: point-independent bounds
  `(1 - sigma) R - A <= Q <= (1 + rho) R + A` between the quantized and the
  real-cost objective, which carry a cut from one cost quantization to
  another (e.g. across `refine_intensity_precision()`).
- `NetworkSimplex.repair_max_nodes` gates warm dual repair by subgraph size:
  subgraphs with more nodes restart cold when the cheap tree patch fails
  (as `warm_violation_limit = 0`); smaller ones keep `warm_violation_limit`.
  On large connected components a repair can cost tens of cold solves;
  measured on 2-D LC-MS alignment once components percolate: 302 s with
  repair vs 16 s gated. The default is 1000 when crash starts are available
  and -1 (disabled) otherwise; see the trash start below.

### Changed

- `NetworkSimplex` defaults to `start=StartMode.Trash` and, when crash starts
  are available, to `repair_max_nodes=1000`. Measured on x86-64-v3 against
  the artificial start with the same warm mode: 2-D deconvolution runs take
  0.48–0.97 of the time, synthetic networks 0.08–1.0, and adding the 1000-node
  repair gate brings large components to 0.01–0.29 (at most 6% slower on small
  ones). AVX-512 and ARM64 showed the same pattern. Without crash starts, the
  same gate was up to 3.5× slower, which is why it stays off against older
  pylmcf. To restore the previous behaviour, pass
  `NetworkSimplex(start=StartMode.Artificial, repair_max_nodes=-1)`.
- With pylmcf's deterministic warm-repair budget (work units instead of wall
  time), a solve sequence no longer depends on timing: previously an interrupt
  during a microsecond-scale repair on a tiny subgraph could trip the budget,
  return a different optimal basis, and steer an outer optimizer to a
  different answer from run to run.

## 1.5.0

### New APIs

- SlopeDP and ConvexSweep now support `WassersteinNetwork.dual_cut()`, with the
  same validated, continuous-supply certificates as NetworkSimplex.
  - `SlopeDP(dual_mode="lazy")` (the default) recovers matching potentials from
    the chain flow only when a cut is requested. `dual_mode="cluster"` retains
    the cluster arrays and recovers cluster potentials instead.
  - ConvexSweep recovers and validates its dual eagerly when solved with
    `solve(..., with_duals=True)`; ordinary solves are unaffected.
  - Subgraph diagnostics: `slope_dp_dual_stats()` and `convex_sweep_dual_stats()`.
- `SlopeDP(storage=...)` and `ConvexSweep(storage=...)` select the breakpoint
  store: `"flat"` (the default), `"tree"` (ordered map) or `"augmented"`
  (persistent split/join treap). Results are identical across stores. See
  [Balanced-tree storage](docs/balanced_tree_storage.md) and
  [Augmented-tree storage](docs/augmented_tree_storage.md).
- `WassersteinNetwork(cost_precision_bits=bits)` and
  `set_cost_precision_bits(bits)` cap the automatic cost scale so that the
  largest relevant integer cost targets `2**bits` (1–52). This opts p=1 into
  cost scaling; an explicit positive `set_cost_scaling(scale)` takes precedence,
  and precision refinement retains the target.
- `get_subgraph(k).convex_sweep_cost_quantization_error()` returns a
  conservative per-unit bound on the ConvexSweep cost quantization error.

### Behavior changes

- ConvexSweep now quantizes geometry rather than individual pair costs: chain
  coordinates are rounded once onto a shared grid and integer convex costs are
  evaluated on it (linear and squared for p=1 and p=2, an integer tangent hull
  otherwise). This preserves the Monge order the sweep relies on; independently
  rounded pair costs could violate it and produce suboptimal matchings.
  ConvexSweep objectives and `scale_factor()` can therefore differ from 1.4.0.
  Unscaled p=1 uses a unit coordinate grid instead of truncating each distance,
  and automatic p=2 scaling budgets against trash escape prices rather than the
  squared span of a chain run. Other solvers keep their existing quantization.
  See [Monge-preserving quantization](docs/monge_quantization.md).

### Fixes

- Build with libc++ and MSVC: the augmented treap no longer uses
  `std::shared_ptr::unique()`, which C++20 removed.

### Validation

- Add regression coverage for ConvexSweep and SlopeDP dual certificates,
  Monge-preserving quantization, cost precision targets and all three
  breakpoint stores, and port the trash reconciliation oracle tests to the
  current API. The test extra now includes `networkx`.

## 1.4.0

### New APIs

- `WassersteinNetwork.dual_cut()` returns an affine lower bound and an upper
  bound for the continuous-supply objective from a solved NetworkSimplex
  network, including LEMON and LinkCut variants. The bounds apply to the
  network's fixed quantized costs; `total_cost()` still returns the rounded
  objective.
- `network.get_subgraph(k).dual_values()` exposes raw integer dual certificates,
  including potentials, reduced costs and bound multipliers.
- `WassersteinNetwork.refine_intensity_precision(factor)` rebuilds a
  floating-intensity network at finer supply resolution. It preserves current
  edge costs, topology, components, isolated peaks and solver configuration.
  Refinement discards the old solution and warm basis; existing cuts must also
  be discarded. A failed refinement leaves the old network usable.

See [Dual cuts with integer network simplex](docs/dual_cuts.md) for certificate
semantics, rounding bounds and precision refinement.

### Fixes and update behavior

- Preserve independent-trash cost cancellation after quantization and when
  updating peak positions, including repeated updates.
- Preserve moved geometry during precision refinement instead of restoring
  original positions or reconstructing topology after threshold crossings.
- Validate target counts, dimensions and peak counts before position updates.
  Missing targets now raise `ValueError` instead of crashing the process;
  shape rejections leave the old solution usable.
- If a position update fails after mutation begins, mark the network and its
  subgraphs invalid and raise an exception instructing the caller to rebuild.
  Subsequent solves, costs, derivatives, flows, cuts and precision refinement
  are rejected. Updates are not rolled back. `build()` restores the original
  distributions; an optimizer recovering its last accepted positions should
  construct a new network from its saved distributions.
- Keep parent networks alive for retained subgraph references, including
  references held across a rebuild.
- Correct chain-solver and distance-metric docstrings and remove deprecated
  iterator-based pytest parameterization.

### Dependencies and validation

- Require `pylmcf >= 1.3.0` for the dual certificate API.
- Add regression coverage for quantized trash costs, precision refinement,
  malformed position updates, partial mutation failures and rebuild recovery,
  plus free-threading concurrency coverage.
