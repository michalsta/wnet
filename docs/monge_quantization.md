# Monge-preserving ConvexSweep quantization

Independently applying `round(S * abs(x-y)**p)` can destroy the Monge
inequality, even though the real power cost is convex. ConvexSweep assumes
Monge order. Production therefore quantizes the geometry and convex cost
function before solving; it does not detect or repair non-Monge matchings.
The non-Monge cycle-correction and alternative-solver experiment prototypes
have been removed. Validation targets the structurally Monge cost model.

For each chain component, reconstruct its nonnegative coordinates from the
chain origin, then round each coordinate once onto a common grid:

| Order | Coordinate denominator | Integer pair cost |
|---|---|---|
| p=1 | S | absolute grid-coordinate difference |
| p=2 | M=sqrt(S) | square of the grid-coordinate difference |
| Other p≥1 | M=sqrt(S) | maximum of integer affine functions of the absolute grid difference |

Automatic p=2 scaling budgets against escape prices, because only profitable
pairs enter the DP, rather than against the squared span of a whole chain run.
Grid coordinates and chain-gap maps retain separate range checks. This avoids
making the coordinate grid coarse just because a component contains distant
peaks. Explicit cost scales and trash quantization retain their existing units;
M need not be an integer for grid coordinates and squared costs to be integers.
Legacy unscaled p=1 uses a unit coordinate grid, rather than truncating
individual distances. Other solvers keep their existing cost quantization.

For fractional powers and other orders, sample tangents to
`S*(q/M)**p`, round their slopes to integers, and choose integer intercepts
from the convex conjugate on the component's distance domain. Build their
upper hull using wide integer comparisons. Pair evaluation selects one hull
line and performs integer arithmetic. The hull includes zero, so costs are
nonnegative and increasing. The even extension is convex; translating it
over ordered integer coordinates gives a Monge matrix. Floating error in
constructing a line can affect approximation accuracy, but cannot invalidate
this structural guarantee. Curvature-adapted construction uses at most
32,768 tangents per component. Linear and squared costs bypass this hull.

The quantizer is cached by geometry and rebuilt after position changes or
network/precision rebuilds. It is independent of supplies, including zero
target weights. Ordinary solves, traced solves, dual recovery, profit windows
and both derivative orientations use precisely the same model. Profit-window
radii are found from the integer cost function, including any flat intervals.
Coordinates must fit the exact integer range of doubles, and evaluated
profitable-window costs must fit the solver integer type. Distant quadratic
costs keep their exact wide-integer definition; the model is never capped or
flattened to fit int64. These pairs are pruned before kernel/certificate cost
evaluation. Overflow of an evaluated cost raises an exception.

The solver and dual certificate are exact **for this quantized model**.
Its objective need not equal the dense solver's independently rounded matrix,
or SlopeDP's additive rounded-gap model. Comparisons must account for those
different quantizations rather than restore their parity by changing matches.

`get_subgraph(k).convex_sweep_cost_quantization_error()` returns a conservative
per-unit error relative to real `abs(x-y)**p` across that component. It combines
one grid step of distance uncertainty, a derivative bound, the hull's maximum
approximation error and a numerical cushion. The hull error is bounded on
each active integer interval by checking its endpoints and the stationary
point of power-minus-line. Multiply the maximum component error by an upper
bound on matched mass to compare objective values. Trash coefficient rounding
and supply rounding require their own allowances. `dual_cut().rounding_error`
continues to describe supply rounding for fixed model costs; it does not
silently absorb a change of cost model.

Validation includes unrestricted subset-assignment oracles, all ordered
four-point Monge inequalities, all three profile stores, zero-supply target
activation, geometry and precision updates, and the two original rounding
regressions. `prototypes/test_monge_cost.cpp` also runs with software 128-bit
arithmetic and UBSan.
