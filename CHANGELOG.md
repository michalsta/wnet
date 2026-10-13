# Release notes

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
