"""
Warm-restart correctness at the wnet layer.

Exercises the full DeconvSolver/WassersteinNetwork -> C++ -> LEMON warm path,
including NetworkSimplex::dualSimplexRepair(), by re-solving the same graph at
a sequence of points (fractions).  Changing the point rescales theoretical
intensities, which changes arc capacities and node supplies between solves
while costs stay fixed -- exactly the scenario warm/dual restart targets.

Oracle: for every point, the warm result (WarmMode.Simple and WarmMode.Dual)
must have *exactly* the same total cost as an independent cold solve
(WarmMode.NONE).  A feasible flow whose cost equals the cold optimum is
optimal, so equal integer total cost across the whole trajectory is a sound
correctness certificate.  We also assert the dual path is actually exercised.
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


def _build(seed, dim, n_targets, size, mode, trash=7):
    rng = np.random.default_rng(seed)
    base = _dist(rng, dim, size)
    targets = [_dist(rng, dim, size) for _ in range(n_targets)]
    ns = NetworkSimplex()
    ns.warm = mode
    net = WassersteinNetwork(base, targets, DistanceMetric.L2, solver=ns)
    net.add_simple_trash(trash)
    net.build()
    return net


def _trajectory(n_targets):
    """A demanding sequence of points: identical re-solve, tiny perturbations,
    large jumps, and boundary (0/1) fractions."""
    base = np.full(n_targets, 0.5)
    pts = [base.copy(), base.copy()]  # identical re-solve
    rng = np.random.default_rng(12345)
    for _ in range(8):
        pts.append(np.clip(base + rng.normal(scale=1e-4, size=n_targets), 0, 1))
    for _ in range(8):
        pts.append(rng.uniform(0.05, 1.0, size=n_targets))  # large jumps
    pts.append(np.full(n_targets, 1.0))  # boundary
    pts.append(np.full(n_targets, 0.05))
    for _ in range(6):
        pts.append(rng.uniform(0.0, 1.0, size=n_targets))
    return pts


@pytest.mark.parametrize("dim", [1, 3])  # 1 -> chain factory, 3 -> dense
@pytest.mark.parametrize("n_targets", [2, 3])
@pytest.mark.parametrize("warm_mode", [
    WarmMode.Simple, WarmMode.Dual, WarmMode.Primal,
    WarmMode.DualRatio, WarmMode.DualGreedy, WarmMode.LinkCut,
])
def test_warm_matches_cold_along_trajectory(dim, n_targets, warm_mode):
    size = 25
    seed = 1000 * dim + 7 * n_targets
    pts = _trajectory(n_targets)

    warm_net = _build(seed, dim, n_targets, size, warm_mode)
    cold_net = _build(seed, dim, n_targets, size, WarmMode.NONE)

    for k, p in enumerate(pts):
        warm_net.solve(p)
        cold_net.solve(p)
        wc = warm_net.total_cost()
        cc = cold_net.total_cost()
        assert wc == cc, (
            f"warm({warm_mode}) != cold at step {k}, point={p}: "
            f"{wc} vs {cc} (dim={dim}, n_targets={n_targets})"
        )

    # Sanity on the C++ counters.
    w = warm_net.wnet.warm_start_count()
    c = warm_net.wnet.cold_start_count()
    d = warm_net.wnet.dual_repair_count()
    pr = warm_net.wnet.primal_repair_count()
    assert w + c + d + pr > 0
    if warm_mode == WarmMode.NONE:
        assert d == 0 and pr == 0
    if warm_mode == WarmMode.Simple:
        assert d == 0 and pr == 0  # Simple never invokes dual/primal repair
    if warm_mode == WarmMode.Dual:
        assert pr == 0  # Dual never invokes the primal path
    if warm_mode == WarmMode.Primal:
        assert d == 0  # Primal never invokes the dual path
    if warm_mode == WarmMode.DualRatio:
        assert pr == 0  # DualRatio never invokes the primal path
    if warm_mode == WarmMode.DualGreedy:
        assert pr == 0  # DualGreedy never invokes the primal path
    if warm_mode == WarmMode.LinkCut:
        # LinkCut is a separate (link-cut-tree) backend with the Simple warm
        # strategy: no dual/primal repair counters.
        assert d == 0 and pr == 0


def test_dual_path_is_actually_exercised():
    """Across a demanding trajectory the Dual mode must hit dualSimplexRepair
    (otherwise this whole suite would be vacuous), and must still be correct."""
    dim, n_targets, size = 3, 3, 30
    pts = _trajectory(n_targets)
    dual = _build(11, dim, n_targets, size, WarmMode.Dual)
    cold = _build(11, dim, n_targets, size, WarmMode.NONE)
    for p in pts:
        dual.solve(p)
        cold.solve(p)
        assert dual.total_cost() == cold.total_cost()
    assert dual.wnet.dual_repair_count() > 0, (
        "dual repair never triggered -- test would be vacuous; "
        "tighten the trajectory"
    )


def test_primal_path_is_actually_exercised():
    """Across a demanding trajectory the Primal mode must hit
    primalSimplexRepair (otherwise this whole suite would be vacuous), must
    never touch the dual path, and must still be correct."""
    dim, n_targets, size = 3, 3, 30
    pts = _trajectory(n_targets)
    primal = _build(11, dim, n_targets, size, WarmMode.Primal)
    cold = _build(11, dim, n_targets, size, WarmMode.NONE)
    for p in pts:
        primal.solve(p)
        cold.solve(p)
        assert primal.total_cost() == cold.total_cost()
    assert primal.wnet.primal_repair_count() > 0, (
        "primal repair never triggered -- test would be vacuous; "
        "tighten the trajectory"
    )
    assert primal.wnet.dual_repair_count() == 0


def test_dualratio_path_is_actually_exercised():
    """DualRatio must hit dualRatioRepair and still be correct."""
    dim, n_targets, size = 3, 3, 30
    pts = _trajectory(n_targets)
    dr = _build(11, dim, n_targets, size, WarmMode.DualRatio)
    cold = _build(11, dim, n_targets, size, WarmMode.NONE)
    for p in pts:
        dr.solve(p)
        cold.solve(p)
        assert dr.total_cost() == cold.total_cost()
    assert dr.wnet.dual_repair_count() > 0, (
        "dualratio repair never triggered -- test would be vacuous; "
        "tighten the trajectory"
    )
    assert dr.wnet.primal_repair_count() == 0


def test_dualgreedy_path_is_actually_exercised():
    """DualGreedy must hit dualGreedyRepair and still be correct."""
    dim, n_targets, size = 3, 3, 30
    pts = _trajectory(n_targets)
    dg = _build(11, dim, n_targets, size, WarmMode.DualGreedy)
    cold = _build(11, dim, n_targets, size, WarmMode.NONE)
    for p in pts:
        dg.solve(p)
        cold.solve(p)
        assert dg.total_cost() == cold.total_cost()
    assert dg.wnet.dual_repair_count() > 0, (
        "dualgreedy repair never triggered -- test would be vacuous; "
        "tighten the trajectory"
    )
    assert dg.wnet.primal_repair_count() == 0


def test_dual_not_worse_than_simple_on_cold_fallbacks():
    """Dual, Primal, DualRatio, and DualGreedy should convert some cold
    fallbacks into repairs (never more cold starts than Simple).  Correctness
    already covered above; this guards the performance intent."""
    dim, n_targets, size = 3, 3, 30
    pts = _trajectory(n_targets)
    simple   = _build(11, dim, n_targets, size, WarmMode.Simple)
    dual     = _build(11, dim, n_targets, size, WarmMode.Dual)
    primal   = _build(11, dim, n_targets, size, WarmMode.Primal)
    dualR    = _build(11, dim, n_targets, size, WarmMode.DualRatio)
    dualG    = _build(11, dim, n_targets, size, WarmMode.DualGreedy)
    for p in pts:
        simple.solve(p)
        dual.solve(p)
        primal.solve(p)
        dualR.solve(p)
        dualG.solve(p)
    assert dual.wnet.cold_start_count()   <= simple.wnet.cold_start_count()
    assert primal.wnet.cold_start_count() <= simple.wnet.cold_start_count()
    assert dualR.wnet.cold_start_count()  <= simple.wnet.cold_start_count()
    assert dualG.wnet.cold_start_count()  <= simple.wnet.cold_start_count()
