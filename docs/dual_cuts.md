# Dual cuts from integer transport solves

`WassersteinNetwork.dual_cut()` constructs an affine lower bound for the
continuous-supply objective using the existing integer transport solve.
NetworkSimplex (LEMON and LinkCut), 1D SlopeDP and ConvexSweep are supported.
NetworkSimplex and SlopeDP reuse their transport state directly. ConvexSweep
uses optional traced DP recovery described below.

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

## ConvexSweep cost quantization

ConvexSweep requires a Monge cost matrix. Independently rounding each real
pair cost can violate that condition, so the solver first rounds all chain
coordinates onto one shared integer grid. At p=1 the cost is the integer
absolute difference; at p=2 it is its exact integer square. Other powers use
an integer convex tangent envelope. No non-Monge correction or geometry-wide
Monge check runs in production. See [Monge quantization](monge_quantization.md).

`total_cost()`, discrete marginal derivatives and the recovered dual all use
this same model. It generally differs from dense independently rounded pair
costs and from SlopeDP's sum of rounded gap costs. The returned cut is exact
for the fixed model, with the usual supply-rounding allowance; that allowance
does **not** include coordinate/cost-model approximation. A per-unit bound
relative to real `distance**p` is available from
`get_subgraph(k).convex_sweep_cost_quantization_error()`.

## ConvexSweep recovery

Use `network.solve(weights, with_duals=True)` with ConvexSweep to recover
certificates eagerly. This selects a boolean-templated sweep kernel; ordinary
solves use the `false` specialization, which retains no traceback records and
does no dual recovery. `dual_cut()` requires a preceding traced solve and
only reads the cached certificate. Position updates preserve the last solve's
choice; precision rebuilds invalidate the solution.

The traced sweep saves predecessor profiles and backtracks an optimal matching.
Complementary slackness then gives tight constraints on matched pairs. Integer
potential propagation enforces these equalities and the remaining matching
inequalities, scanning profitable position windows on demand. There is no
auxiliary simplex solve and no stored graph of all profitable pairs.
Zero-capacity positions remain in the constraints so cuts stay valid when
zero mixture weights grow. Separate discrete marginal derivatives are not
used as a common matching dual.

Recovered coefficients lie in `[-tau, 0]`. Every profitable pair inequality,
positive-match equality, capacity constraint, and exact primal/dual objective
agreement is checked with wide integer arithmetic before caching the result.
A disagreement raises an error. The usual trash bracket and supply-rounding
bound then apply.

Traceback storage is proportional to the sum of saved piecewise-linear profile
and profit-prefix sizes across events, potentially quadratic for wide windows.
It is released after recovery. Potential propagation retains `O(N + M)` state
for `N` position blocks and `M` matched pairs, and can scan windows repeatedly;
it does not guarantee linear recovery time. Only `O(N)` dual coefficients
remain cached for cut queries.

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
