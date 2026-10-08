# Dual cuts with integer network simplex

`WassersteinNetwork.dual_cut()` constructs an affine lower bound for the
continuous-supply objective using the existing integer NetworkSimplex solve.
LEMON and LinkCut variants are supported. It does not run an LP transport
solver or a residual shortest-path search.

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

Raw per-subgraph certificates are exposed through
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

## Supply error and precision

For dropped rounding mass dE,dT, a conservative bound is
`(Ce + tau)*dE + (Ct + tau)*dT`. It follows from the matching dual's bounded
multipliers and A's bounded slopes. The reported `rounding_error` includes
this quantity in real units. Tests compare both global cuts and upper bounds
against independently solved continuous matching LPs, including coarse
supplies, all trash models, sparse/disconnected graphs and chain semantics.

For floating-intensity inputs, `network.refine_intensity_precision(factor)`
rebuilds the same graph/backend
at finer supply resolution and selects a safe cost scale again. It invalidates
the old solution and warm basis. Existing cuts must also be discarded because
their cost coefficients can change. Failed rebuilds preserve the old network.
LinkCut cost scaling additionally reserves headroom for its artificial-cost
and path-potential arithmetic, rather than checking only the final bill.

The production cutting-plane integration in wnetdeconv uses these bounds,
refines supply precision when rounding limits progress, and rebuilds its model
after refinement. Backends without this certificate retain a clearly marked
heuristic search; improvement or a heuristic gap alone is not certified success.
