# Release notes

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
