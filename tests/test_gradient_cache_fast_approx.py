"""
Value-exact DerivContext cache + opt-in fast_approx (dual-pi) gradient.

Two independent features added on top of the residual-marginal gradient:

1. Caching: signal/spectrum derivatives memoize a per-solve context, rebuilt
   only when the solution changes (a solve bumps an internal version).  This
   MUST be value-exact: a cached query is bit-identical to a fresh recompute.
   Oracle = a second network that only ever solves the queried point (so it
   never reuses a cached context).

2. fast_approx: the pure dual-potential difference instead of the residual
   shortest-path search.  Faster on dense (NetworkSimplex/Dijkstra)
   subgraphs, but a DIFFERENT, basis-dependent gradient (a lower bound on the
   true marginal).  For chain subgraphs there are no NS potentials, so it
   falls back to the exact chain search and must equal the exact result.
"""

import numpy as np
import pytest

from wnet import Distribution, WassersteinNetwork
from wnet.distances import DistanceMetric
from wnet.wnet_cpp import NetworkSimplex, WarmMode


def _dist(rng, dim, size, pos_range=20.0, int_range=5.0):
    data = rng.uniform(size=(dim, size)) * pos_range
    inten = rng.uniform(size=(size,)) * int_range
    return Distribution(data, inten)


def _build(seed, dim, n_targets, size, trash=7, mode=WarmMode.Dual):
    rng = np.random.default_rng(seed)
    base = _dist(rng, dim, size)
    targets = [_dist(rng, dim, size) for _ in range(n_targets)]
    ns = NetworkSimplex()
    ns.warm = mode
    net = WassersteinNetwork(base, targets, DistanceMetric.L2, solver=ns)
    net.add_simple_trash(trash)
    net.build()
    return net


def _traj(n_targets):
    base = np.full(n_targets, 0.5)
    rng = np.random.default_rng(2024)
    pts = [base.copy(), base.copy()]  # identical re-solve
    for _ in range(6):
        pts.append(np.clip(base + rng.normal(scale=1e-4, size=n_targets), 0, 1))
    for _ in range(6):
        pts.append(rng.uniform(0.05, 1.0, size=n_targets))
    return pts


@pytest.mark.parametrize("dim", [1, 3])  # 1 -> chain, 3 -> dense
@pytest.mark.parametrize("n_targets", [2, 3])
def test_cache_is_value_exact_vs_fresh_oracle(dim, n_targets):
    """Cached exact gradient/derivs along a trajectory must equal an oracle
    network that solves each point in isolation (no cross-solve cache reuse)."""
    # WarmMode.NONE: every solve is a cold solve, so the basis (hence the
    # exact residual gradient) is a deterministic function of the point.
    # That makes the fresh-network oracle valid and isolates the CACHE
    # (warm chaining would land on a different but equally-optimal basis,
    # a separate already-documented degenerate-dual effect).
    size = 22
    seed = 100 * dim + n_targets
    pts = _traj(n_targets)

    net = _build(seed, dim, n_targets, size, mode=WarmMode.NONE)
    for p in pts:
        net.solve(p)
        # Two queries on the SAME solution -> 2nd is a pure cache hit.
        g1 = net.spectrum_proportion_derivatives()
        g2 = net.spectrum_proportion_derivatives()
        assert np.array_equal(g1, g2)
        s1 = net.signal_part_derivatives()
        s2 = net.signal_part_derivatives()
        assert s1 == s2
        # Interleave the other (fast) slot: must not corrupt the exact slot.
        net.spectrum_proportion_derivatives_fast_approx()
        net.signal_part_derivatives_fast_approx()
        assert np.array_equal(net.spectrum_proportion_derivatives(), g1)
        assert net.signal_part_derivatives() == s1

        # Oracle: fresh network that only solves THIS point (cold).
        oracle = _build(seed, dim, n_targets, size, mode=WarmMode.NONE)
        oracle.solve(p)
        assert np.array_equal(g1, oracle.spectrum_proportion_derivatives())
        assert s1 == oracle.signal_part_derivatives()


@pytest.mark.parametrize("dim", [1, 3])
def test_fast_approx_shape_finite_and_chain_equivalence(dim):
    n_targets, size = 3, 25
    net = _build(7 * dim, dim, n_targets, size)
    net.solve(np.full(n_targets, 0.5))

    exact = net.spectrum_proportion_derivatives()
    approx = net.spectrum_proportion_derivatives_fast_approx()
    assert approx.shape == exact.shape
    assert np.all(np.isfinite(approx))

    sp_exact = net.signal_part_derivatives()
    sp_approx = net.signal_part_derivatives_fast_approx()
    assert set(sp_approx.keys()) == set(sp_exact.keys())

    if dim == 1:
        # Chain subgraphs have no NS potentials -> fast_approx must fall back
        # to the exact chain search and be value-identical.
        assert np.array_equal(approx, exact)
        assert sp_approx == sp_exact


def test_cache_invalidated_by_resolve():
    """A re-solve at a different point must invalidate the cache (gradient
    tracks the new solution, equals a fresh oracle for that point)."""
    net = _build(42, 3, 3, 25, mode=WarmMode.NONE)
    net.solve(np.array([0.3, 0.5, 0.7]))
    _ = net.spectrum_proportion_derivatives()  # populate cache slot
    net.solve(np.array([0.9, 0.1, 0.4]))
    g_b = net.spectrum_proportion_derivatives()
    oracle = _build(42, 3, 3, 25, mode=WarmMode.NONE)
    oracle.solve(np.array([0.9, 0.1, 0.4]))
    assert np.array_equal(g_b, oracle.spectrum_proportion_derivatives())
