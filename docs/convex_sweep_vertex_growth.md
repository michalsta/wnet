# ConvexSweep: per-event vertex growth (measured 2026-10-16)

`docs/wp_chain_design.md` targets O((P_e + P_t) · W) for the sweep, W = 2R/Δ
grid steps per window, on the premise that value functions have
O(blocks-in-window) vertices. Measured on dense profile data (p = 2, simple
trash τ, R = √τ), they do not: the mean vertex count per event grows as W².

Traced solves (`solve(..., with_duals=True)`, `convex_sweep_dual_stats()`),
synthetic NMR profile, 2 spectra:

| n    | τ        | W   | solve (s) | vertices / event | ÷ W  |
|------|----------|-----|-----------|------------------|------|
| 2000 | 0.003125 | 22  | 0.093     | 121.7            | 5.4  |
| 2000 | 0.0125   | 45  | 0.287     | 409.4            | 9.2  |
| 2000 | 0.05     | 89  | 1.047     | 1542.3           | 17.2 |
| 2000 | 0.2      | 179 | 3.920     | 5873.6           | 32.8 |
| 500  | 0.05     | 22  | 0.036     | 131.3            | 5.9  |
| 1000 | 0.05     | 45  | 0.136     | 421.9            | 9.4  |
| 4000 | 0.05     | 179 | 7.726     | 5870.4           | 32.8 |

Vertices per event ≈ 0.18 W², depending on W only (n = 4000, τ = 0.05 and
n = 2000, τ = 0.2 coincide). Per-event work is O(|V|), so a solve is
O(n · W²); for a fixed physical radius W ∝ n, hence the observed ~n³.

Slimming does not help (`prototypes/sweep_vertex_probe.cpp` with temporary
hooks in `sweep_step()`, not kept; n = 2000, W ≈ 89):

- exact: `normalize()` after every step (merges collinear vertices only)
  removes nothing — 1400.7 vertices/event either way, identical profit;
- approximate: dropping vertices within ε integer units of the interpolant
  removes nothing at ε ≤ 100 and 1% at ε = 10⁴; ε = 10⁶ breaks the
  traceback's adjacent-integer invariants (`Invalid sweep append traceback`).

The breakpoints are genuine: crossing insertions of the non-concave value
function over unit counts add O(W) per event and live for ~W events. The
per-vertex operations (binary searches in `add_pl` / `window_sub`) are not
the quadratic factor. A real fix needs a different DP formulation (e.g. one
exploiting the Monge structure of p = 2 over position blocks instead of
materialising V over unit counts), not a representation tweak.
