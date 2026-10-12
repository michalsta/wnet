# Dual cuts from integer transport solves

`WassersteinNetwork.dual_cut()` constructs an affine lower bound for the
continuous-supply objective using the existing integer transport solve.
NetworkSimplex (LEMON and LinkCut), 1D SlopeDP and ConvexSweep are supported.
NetworkSimplex and SlopeDP reuse their transport state directly. ConvexSweep
uses the auxiliary certificate recovery described below.

```python
network.solve(weights)
cut = network.dual_cut()
lower_at_y = cut["intercept"] + cut["gradient"] @ y
upper_at_weights = cut["upper_bound"]
rounded_cost = cut["rounded_cost"]
```

These bounds refer to the network's **fixed quantized edge/trash costs** with
continuous peak supplies. The rounded objective remains available unchanged
through `total_cost()`. The upper bound includes an explicit supply-rounding
error plus a floating-point accumulation cushion. Cost-coefficient rounding is
not silently claimed to be exact original real-coefficient optimization.

NetworkSimplex raw per-subgraph certificates are exposed through
`network.get_subgraph(k).dual_values()`: potentials, reduced costs, signed
lower-bound multipliers and signed upper-bound multipliers, as int64 arrays in
subgraph node/arc order. Products and sums require checked/wide arithmetic;
raw potentials can contain artificial-root offsets.

## Why the cut is global

Let E and T be total empirical and theoretical mass. The objective can be
written as `A(E,T) + matching_profit`, where the matching problem has costs
`distance - tau`, peak-capacity bounds, and optional unmatched mass:

| Trash model | A(E,T) | tau |
|---|---|---|
| Independent | Ce E + Ct T | Ce + Ct |
| Simple | C max(E,T) | C |
| Both, Ce ≤ Ct | Ce E + Ct max(T-E,0) | Ce |
| Both, Ce > Ct | Ct T + Ce max(E-T,0) | Ct |
| Empirical only | Ce E | Ce |
| Theoretical only | Ct T | Ct |

The matching dual has peak multipliers alpha,beta ≤ 0, subject to
`alpha_i + beta_j <= distance_ij - tau`. One **common feasible dual** therefore
gives a lower bound `sum(alpha_i E_i) + sum(beta_j T_j)` for all supplies.
Since `T_j = intensity_j * weight[spectrum_j]`, this is affine in the weights.
Adding a supporting affine piece of convex A gives the returned cut. Global
A handles the annihilation correction across components and isolated peaks.

The implementation reads the solved simplex potentials, cancels their common
source offset in checked integer arithmetic, chooses a compatible potential
shift at degeneracy, and clips matching multipliers to [-tau,0]. Nonnegative
distances make that clipping preserve dual feasibility. Redundant matching-arc
capacities are removed for simplex backends: empirical/theoretical anchor
capacities already bound every matching flow. This eliminates arbitrary
matching-cap multipliers at saturation and per-pair capacity updates during
warm solves. Other backends retain their capacity representation.

Scratch buffers and anchor/trash arc indices are retained per component.
Cut construction accumulates coefficients directly in C++; full dual arrays
are only materialized when explicitly requested by Python.

## ConvexSweep recovery

The sweep's separate discrete marginal derivatives do not constitute a
common matching dual. On the first cut request after a solve, ConvexSweep
constructs an auxiliary partial-matching circulation and runs integer network
simplex. It includes only pairs whose quantized cost is less than the matching
reward `tau`; all other matching constraints are satisfied automatically by
nonpositive dual coefficients. Zero-capacity theoretical positions remain in
the auxiliary problem so cuts remain valid when zero mixture weights grow.

The recovered coefficients lie in `[-tau, 0]`. Every retained pair constraint
is checked, and the auxiliary primal, dual, and sweep objectives must agree
in exact 128-bit arithmetic before a certificate is returned. A disagreement
raises an error rather than returning a certificate. The usual trash bracket
and supply-rounding bound then apply.

Recovery is optional and cached for one solution epoch. A new solve, position
update, or precision rebuild invalidates it. Cost-only solves and marginal
queries do not construct the auxiliary graph. With `N` position blocks and
`W` blocks in the profitable window, recovery can retain `O(NW)` arcs (up to
quadratic for a wide window), plus the cost of a simplex solve. Thus certified
optimization currently sacrifices the sweep's small working-memory advantage
during recovery. Direct dual recovery from the sweep remains future work.

## Supply error and precision

For dropped rounding mass dE,dT, a conservative bound is
`(Ce + tau)*dE + (Ct + tau)*dT`. It follows from the matching dual's bounded
multipliers and A's bounded slopes. The reported `rounding_error` includes
this quantity in real units. Tests compare both global cuts and upper bounds
against independently solved continuous matching LPs, including coarse
supplies, all trash models, sparse/disconnected graphs and chain semantics.

For floating-intensity inputs, `network.refine_intensity_precision(factor)`
rebuilds the same graph/backend
at finer supply resolution and selects a safe cost scale again. It preserves
current edge costs after position updates, along with the existing edges,
components and isolated peaks, even if moves crossed a distance threshold.
It invalidates the old solution and warm basis. Existing cuts must also be discarded because
their cost coefficients can change. Failed rebuilds preserve the old network.
LinkCut cost scaling additionally reserves headroom for its artificial-cost
and path-potential arithmetic, rather than checking only the final bill.

Position updates validate target counts, dimensions and peak counts before
changing costs. A shape rejection leaves the old solution usable. Other
update failures, such as crossing peaks in a later chain component or cost
overflow after an earlier component has been updated, can leave partial
changes. The network is then marked invalid and raises an exception with a
rebuild instruction. Further solves, costs, derivatives, flows, cuts and
precision refinement are rejected, including through retained subgraphs.
Updates are not rolled back. `build()` recreates the original distributions;
an optimizer should retain its last accepted distributions and construct a
new network from them before retrying a rejected step.

The production cutting-plane integration in wnetdeconv uses these bounds,
refines supply precision when rounding limits progress, and rebuilds its model
after refinement. Backends without this certificate retain a clearly marked
heuristic search; improvement or a heuristic gap alone is not certified success.

## Direct recovery from 1D DP

`SlopeDP(dual_mode="lazy")` (the default) recovers matching potentials from
expanded DP flows only when `dual_cut()` is called. Ordinary solves retain no
additional cluster arrays and perform no dual recovery or validation.
`SlopeDP(dual_mode="cluster")` instead moves the five existing cluster capacity
and matched-flow arrays into solver state, retaining about 40 bytes per cluster.
It recovers cluster potentials and expands them to peaks when a cut is requested.
Both modes validate capacity bounds, flow conservation, complementary slackness,
chain dual feasibility and exact integer primal/dual objective equality. The
validated potentials are cached until the next solve or position update.
Recovery and validation take linear time in the chain size; neither mode solves
a second transport problem. Both use the same continuous-supply rounding bounds
as NetworkSimplex. Raw `dual_values()` remains a NetworkSimplex API.

```python
from wnet.wnet_cpp import SlopeDP
network = WassersteinNetwork(empirical, components, distance,
                             split_distance=10, solver=SlopeDP(dual_mode="cluster"))
network.add_simple_trash(3)
network.build()
network.solve([0.4, 0.6])
cut = network.dual_cut()
```

Configurations are copied when the graph is built. Choose the mode before
building, or rebuild after changing it. `slope_dp_dual_stats()` on a subgraph
returns `(recoveries, retained_clusters, potential_count)` for diagnostics.
