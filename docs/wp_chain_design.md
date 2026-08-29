# Chain-native solver for W_p, p > 1 — design note

Status: design only, no implementation. Written before any code, so that the
correctness-critical claims are stated (and separated from hypotheses) up
front.

## 1. Problem

For 1-D data with `p == 1`, wnet solves the inner transport problem with the
chain factory and the SlopeDP backend: O(K log K) per solve on K sorted
positions, no dense arcs. For `p != 1` (in particular W2, cost = distance²)
only the dense factory exists: O(m·n) matching arcs within `max_distance`.
On profile NMR this is the difference between milliseconds and hours:

* perfumes (12 449 pts/spectrum, 4 components): ~2.3e7 dense arcs at the
  W2-effective cap, ~6 GB, and NetworkSimplex-driven outer optimization runs
  for hours per deconvolution.
* overlap (131 072 pts): ~1e9 arcs at W1 caps — does not fit in 122 GB.

Goal: an exact chain-native solver for strictly convex costs c(d) = d^p,
p > 1 (and, since nothing below uses the exponent, any strictly convex
increasing c), supporting the existing trash models, proportion scaling,
integer quantization, and exact marginals for the outer optimizer.

Note the arcs themselves are not the bottleneck — the windowed dense graph is
buildable for medium inputs. The bottleneck is solving it with a generic MCF
algorithm thousands of times inside L-BFGS-B/SLSQP. The win must come from an
algorithm that exploits 1-D structure, as SlopeDP does for p = 1.

## 2. Why SlopeDP does not generalize

SlopeDP's invariant is that transport cost is additive over chain gaps: a
unit in transit pays each gap `gap_cost` regardless of origin. Hence the
whole state per gap is one scalar (net cumulative flux) and the value
function is convex piecewise linear in it.

For p > 1, a unit that has traveled t and crosses a gap of width a pays
c(t+a) − c(t), which depends on t. Units in the same gap owe different
amounts depending on where they boarded. The scalar state is insufficient in
principle, not merely inconvenient; no per-gap cost decomposition of W_p^p
exists for p > 1. (The convex-piecewise-quadratic generalization of the
breakpoint calculus is coherent — derivatives become piecewise linear and
min-plus convolution is still addition of inverse derivatives — but there is
no gap-local transition to feed it.)

Consequently this design abandons hop arcs entirely: the proposed solver is a
sweep over sorted positions, computing pairwise costs c(|x − y|) directly,
exactly as the dense factory prices them. This also means cost quantization
is per pair, identical to dense — bit-parity with the dense factory is
achievable (unlike W1 chains, which sum quantized gaps).

## 3. Structural facts (proved)

Let c: R -> R be convex with c(t) = c(|t|), strictly convex for p > 1.
Positions sorted ascending on the line.

**(F1) Quadrangle / Monge inequality.** For x1 < x2 and y1 < y2:
`c(y1−x1) + c(y2−x2) <= c(y2−x1) + c(y1−x2)`, strict for strictly convex c
unless degenerate. Proof: the two sums have equal argument totals and the
crossed pair majorizes the uncrossed pair (y2−x1 is the extreme value);
convexity is Schur-convexity on two points.

**(F2) Monotone (non-crossing) optimal couplings.** By (F1), any optimal
(partial) coupling can be uncrossed without cost increase; for strictly
convex c every optimal coupling of the *transported* units is monotone:
sorting matched empirical units and matched theoretical units, the k-th
matches the k-th.

**(F3) One-sided pending.** In a monotone matching, a left-to-right sweep
never simultaneously holds an unmatched-yet-to-match-right empirical unit
and an unmatched-yet-to-match-right theoretical unit: the two eventual pairs
would cross. So at any sweep point the "open" interface consists of pending
units of one side only.

**(F4) Selection is not nested.** Which units end up matched depends on
future capacity, not only on counts: with pending empirical units x1 < x2
and a single future theoretical unit at y, the optimum matches x2 (nearer);
with two future units y1 < y2 and both matched, monotonicity forces
x1 -> y1, x2 -> y2. Hence a DP whose state is only "number pending" with a
fixed pending identity is wrong; the DP must range over explicit prefixes of
both sides (Section 5) or prove a stronger exchange property.

**(F5) Effective radius.** With the trash reduction of Section 4, a match at
distance d is profitable only if c(d) <= tau, i.e. d <= R := c^{-1}(tau).
For c = d²: R = sqrt(tau). Everything beyond R can be pruned, and subgraph
decomposition may split components at gaps > R exactly (no hop arcs exist to
bridge them, and no profitable pair spans them).

## 4. Trash models reduce to a constant-tau profit kernel

Identical to the SlopeDP treatment (`trash_of(M)` in `_slope_dp_solve`):

* **Independent trash**: trash_of(M) = C_exp(E−M) + C_theo(T−M), exactly
  affine; tau = C_exp + C_theo. Shifted convention: solver reports
  `transport − tau·M`; `total_cost()` adds the bracket E·C_exp + T·C_theo.
* **Annihilating asymmetric / simple trash**: trash_of(M) is the price-sorted
  channel greedy; affine in M for fixed (E, T, price order), with
  tau = trash_of(0) − trash_of(1), constant per solve. Same as SlopeDP today.

In all cases the kernel is:

> **Maximum-profit monotone partial matching on a line.** Given sorted
> empirical unit positions and sorted theoretical unit positions (with
> integer multiplicities), choose a monotone partial matching maximizing
> `sum over pairs of (tau − c(d))`. Unmatched units cost nothing here (their
> price sits in trash_of(0) / the bracket). Only pairs with d <= R matter.

Total cost = trash_of(0) − (kernel profit) + transport bookkeeping folded as
in SlopeDP; per-arc/per-pair flows recovered from the matching itself.

## 5. Algorithms

### 5.1 Baseline: windowed alignment DP (exact, the fallback)

Unit-level formulation: DP[i][j] = max profit over the first i empirical and
first j theoretical units; transitions skip-emp, skip-theo, or match
(profit tau − c(d(e_i, t_j))). Correct by (F2)/(F4). O(N·M) at unit
granularity is infeasible (units are quantized intensities, ~1e6+).

Collapse to position blocks: co-located units form blocks; matching k units
between block pair (a, b) has profit k·(tau − c(d_ab)), linear in k. The DP
becomes a windowed transportation problem over position blocks: P_e × P_t
block pairs restricted to |d| <= R, i.e. per-block window W ≈ 2R/Δ grid
steps. Within a window the block-level DP with integer flow counts is an
instance of transportation with an inverse-Monge profit matrix (F1).

* Hypothesis H1: with (F1), the block-level DP admits a greedy/NW-corner
  inner structure (Hoffman's theorem solves the *balanced* Monge
  transportation greedily; the partial/trash variant needs the skip options,
  which break global Monge — see risk R1). Target complexity O((P_e+P_t)·W).
* Fallback if H1 fails: windowed unit-collapse DP with concave count
  profiles, or plain windowed dense NetworkSimplex (still wins big on
  memory; loses the speed goal).

Workload arithmetic (perfumes, W2 with squared trash costs, tau ≈ 0.111,
R ≈ 0.333, Δ ≈ 9.6e-4): W ≈ 690, P ≈ 12·10³ → ~8.6e6 block-pair visits,
trivially fast if H1 holds. overlap (Δ ≈ 1.2e-4, raw costs tau ≈ 0.47,
R ≈ 0.69): W ≈ 11·10³, P ≈ 1.3·10⁵ → ~1.4e9 visits — near the edge; squared
trash costs or Monge speedups (5.2) bring it down.

### 5.2 Monge speedups

The alignment DP under (F1) satisfies the quadrangle inequality, so
Knuth-style monotonicity of argmins / SMAWK row-minima apply to the match
decisions. Expected effect: replace the W factor by log W in the inner
optimization. This is an optimization layer over 5.1, not a correctness
requirement.

### 5.3 Balanced core

When trash never binds (all mass matched — rare in our data), the optimum is
the quantile coupling: O(P) after sorting, for any convex c. Useful as a
special case and as a test oracle.

## 6. Marginals

Mirror of today's `dist_src`/`dist_sink` structure, in DP form:

* Forward pass F[·] (prefix optimal values) and backward pass B[·] (suffix
  optimal values) over the block DP interfaces.
* Shifted marginal of one extra theoretical unit at block b =
  `max(0-option, best over windowed empirical blocks e of
  F[left interface] + (tau − c(d_eb)) + B[right interface]) − OPT`,
  where the 0-option is the phantom (unfilled) case. O(W) per block,
  O(P·W) per gradient evaluation, aggregated per spectrum weighted by real
  intensities exactly as `_spectrum_proportion_derivatives_impl` does.
* The independent-trash lesson from the W1 chain applies verbatim: marginals
  must be evaluated in the shifted problem (bracket added outside), and both
  "augment" and "rearrange" alternatives must be considered — the
  min(dist_src, dist_sink) dispatch has its analogue in taking the best over
  both the phantom and the re-matching completions. FD tests are mandatory.

## 7. Quantization and exactness

* Costs: quantize per evaluated pair, `quantize_cost(c(d_real), scale,
  p_is_one=false)` — the same function the dense factory applies per arc.
  Bit-parity with dense is therefore an achievable test criterion (choose
  grid positions and binary-exact costs in tests, as
  `test_independent_trash.py` does).
* p = 2 on rational grids: d² stays exactly representable (grid step 2^-k
  gives d² on a 2^-2k grid); integer supplies as today.
* Fractional p (e.g. 1.5): c(d) is irrational for almost all d; exactness
  degrades to quantization tolerance by principle, not implementation. State
  it in the docs; tests use tolerances, not equality.
* Accumulators: profits are bounded by tau·min(E,T); reuse the existing
  int64 budget machinery (`pick_cost_scale`, flow budget) unchanged.

## 8. Integration

* New solver config (e.g. `ConvexSweep{}`) valid only for 1-D subgraphs and
  p > 1; the factory builds no matching and no chain arcs for it — only the
  src/sink/trash skeleton, like SlopeDP's fabricated flows. Matched pairs are
  kept as a sparse list (matched pairs <= min(E, T) units, typically far
  fewer blocks); arc-level flow introspection stays a dense-factory feature.
* Wrapper: `_decide_chain`-analogous gate; `split_distance` semantics with
  the exactness condition split >= R = c^{-1}(tau) (F5). wnetdeconv would
  route p != 1 one-dimensional solvers here by default once proven.
* Subgraph decomposition: split at gaps > R; isolated nodes keep the
  existing network-level pricing. Both are exact by (F5) and by linearity of
  the trash terms.

## 9. Test plan

1. Kernel unit tests: hand toys (threshold at c(d) = tau; F4's non-nested
   selection example as an explicit regression).
2. Fuzz vs dense factory at p = 2: grid positions, binary-exact costs,
   points on the supply grid (cross-factory isolated-node convention,
   as in `test_chain_fuzz_matches_dense`), totals and gradients.
3. FD gradient checks at fine intensity scale (staircase-aware tolerances).
4. p = 3 (integer-exact) and p = 1.5 (tolerance) spot checks vs dense.
5. Balanced-core oracle: quantile coupling equality when trash is priced
   prohibitively high.
6. Benchmarks: perfumes W2 end-to-end vs the dense runs; overlap W2
   feasibility (currently impossible).

## 10. Risks / open questions

* **R1 (main):** H1 — the greedy/Monge inner structure of the windowed
  partial transportation. Hoffman covers the balanced case; the skip/trash
  options need either a proof (likely via adding boundary dummy rows and a
  profit threshold argument) or the 5.1 fallback. Prototype first.
* **R2:** marginal exactness proof for the prefix/suffix construction at
  degenerate optima (ties). The W1 experience says: prove or fuzz against
  dense residual marginals, not against intuition.
* **R3:** annihilating trash with channel caps hitting zero mid-range
  (tau affine only while both channels have capacity) — same edge SlopeDP
  already navigates; port its guards.
* **R4:** performance of O(P·W) on overlap-sized inputs without 5.2; measure
  before optimizing.

## 11. Staging

1. Python prototype of the block DP kernel (no speedups) vs dense NS on
   toys and perfumes-scale data — validates H1 or falls back.
2. C++ implementation behind `ConvexSweep{}`, totals + flows only.
3. Marginals (forward/backward passes) + FD/fuzz test battery.
4. Monge speedups if R4 demands them.
5. wnetdeconv routing + benchmark reruns (perfumes/overlap W2).
