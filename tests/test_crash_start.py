"""NetworkSimplex StartMode.Trash (crash start from the all-trash flow) must not change any answer.

Every combination of trash model, factory (dense / 1-D chain), warm mode and
point sequence is solved twice, once per start mode, and the total cost and
the exact derivatives must agree exactly.
"""
import numpy as np
import pytest

from wnet import Distribution, WassersteinNetwork
from wnet.distances import DistanceMetric
from wnet.wnet_cpp import NetworkSimplex, StartMode, WarmMode
import wnet.wnet_cpp as _wnet_cpp

# Built against a pylmcf without crashRun (e.g. the 1.3.0 wheel), StartMode.Trash
# falls back to the artificial start: answers are identical, counters stay 0.
HAS_CRASH_START = getattr(_wnet_cpp, "HAS_CRASH_START", False)

MODELS = ["simple", "both", "independent", "experimental", "theoretical"]


def build(model, start, warm, seed, chain, dim=2, n_emp=40, n_theo=30, k=3):
    rng = np.random.default_rng(seed)
    d = 1 if chain else dim
    empirical = Distribution(rng.uniform(0, 10, (d, n_emp)), rng.uniform(0.1, 2, n_emp))
    theoretical = [Distribution(rng.uniform(0, 10, (d, n_theo)), rng.uniform(0.1, 2, n_theo))
                   for _ in range(k)]
    solver = NetworkSimplex()
    solver.warm = warm
    solver.start = start
    kwargs = {"split_distance": 30.0} if chain else {"max_distance": 3.0}
    g = WassersteinNetwork(empirical, theoretical, DistanceMetric.L1, solver=solver,
                           force_dense_1d=not chain, round_max_distance=False, **kwargs)
    g.set_cost_scaling(0)
    if model == "simple":
        g.add_simple_trash(2.0)
    elif model == "independent":
        g.add_independent_asymmetric_trash(1.0, 1.5)
    elif model == "both":
        g.add_experimental_trash(1.0)
        g.add_theoretical_trash(1.5)
    elif model == "experimental":
        g.add_experimental_trash(1.0)
    else:
        g.add_theoretical_trash(1.5)
    g.build()
    return g


def points(seed, k=3, n=8):
    rng = np.random.default_rng(1000 + seed)
    return [rng.uniform(0, 2, k) for _ in range(n)]


ALL_WARM = [WarmMode.NONE, WarmMode.Simple, WarmMode.Dual, WarmMode.Primal, WarmMode.DualRatio,
            WarmMode.DualGreedy]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("chain", [False, True])
@pytest.mark.parametrize("warm", ALL_WARM)
@pytest.mark.parametrize("seed", range(4))
def test_trash_start_matches_artificial(model, chain, warm, seed):
    if chain and model == "independent":
        pytest.skip("independent trash on the chain needs SlopeDP")
    a = build(model, StartMode.Artificial, warm, seed, chain)
    b = build(model, StartMode.Trash, warm, seed, chain)
    for w in points(seed):
        a.solve(list(w))
        b.solve(list(w))
        assert a.total_cost() == b.total_cost()
        np.testing.assert_array_equal(a.spectrum_proportion_derivatives(),
                                      b.spectrum_proportion_derivatives())


@pytest.mark.parametrize("model", MODELS)
def test_trash_start_runtime_switch(model):
    """set_warm_policy() can switch the start mode between solves."""
    a = build(model, StartMode.Artificial, WarmMode.NONE, 7, False)
    b = build(model, StartMode.Artificial, WarmMode.NONE, 7, False)
    p = NetworkSimplex()
    p.warm = WarmMode.NONE
    p.start = StartMode.Trash
    b.set_warm_policy(p)
    for w in points(7):
        a.solve(list(w))
        b.solve(list(w))
        assert a.total_cost() == b.total_cost()


def test_trash_start_high_dimension():
    a = build("independent", StartMode.Artificial, WarmMode.DualRatio, 3, False, dim=6, n_emp=80)
    b = build("independent", StartMode.Trash, WarmMode.DualRatio, 3, False, dim=6, n_emp=80)
    for w in points(3):
        a.solve(list(w))
        b.solve(list(w))
        assert a.total_cost() == b.total_cost()


@pytest.mark.parametrize("warm", ALL_WARM)
def test_crash_start_counters(warm):
    a = build("both", StartMode.Artificial, warm, 5, False)
    b = build("both", StartMode.Trash, warm, 5, False)
    pts = points(5)
    for w in pts:
        a.solve(list(w))
        b.solve(list(w))
    assert a.crash_start_count() == 0
    if not HAS_CRASH_START:
        assert b.crash_start_count() == 0
        return
    assert b.crash_start_count() >= 1
    if warm == WarmMode.NONE:
        # every solve of every subgraph starts cold, hence from the all-trash flow
        assert b.crash_start_count() == len(pts) * b.no_subgraphs()
    assert b.warm_start_count() >= 0 and b.cold_start_count() >= 0


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("warm", [WarmMode.NONE, WarmMode.Dual, WarmMode.DualRatio])
def test_trash_start_position_updates(model, warm):
    """Cost changes (update_positions_and_solve) after crash-started solves stay exact."""
    from wnet import Distribution
    a = build(model, StartMode.Artificial, warm, 11, False)
    b = build(model, StartMode.Trash, warm, 11, False)
    a.solve([1.0, 0.7, 1.3])
    b.solve([1.0, 0.7, 1.3])
    rng = np.random.default_rng(99)
    base_rng = np.random.default_rng(11)
    pos = base_rng.uniform(0, 10, (2, 40))
    ints = base_rng.uniform(0.1, 2, 40)
    theo = [Distribution(base_rng.uniform(0, 10, (2, 30)), base_rng.uniform(0.1, 2, 30)) for _ in range(3)]
    for step in range(4):
        moved = Distribution(pos + rng.normal(0, 0.3, pos.shape), ints)
        a.update_positions_and_solve(moved, theo)
        b.update_positions_and_solve(moved, theo)
        assert a.total_cost() == b.total_cost()


def test_linkcut_ignores_start_mode():
    a = build("simple", StartMode.Artificial, WarmMode.LinkCut, 2, False)
    b = build("simple", StartMode.Trash, WarmMode.LinkCut, 2, False)
    for w in points(2):
        a.solve(list(w))
        b.solve(list(w))
        assert a.total_cost() == b.total_cost()
    assert b.crash_start_count() == 0


def test_network_simplex_keyword_constructor():
    import wnet
    s = wnet.NetworkSimplex(start=wnet.StartMode.Trash, warm=wnet.WarmMode.NONE,
                            pivot=wnet.NSPivotRule.FIRST_ELIGIBLE, repair_max_nodes=17,
                            warm_violation_limit=3, warm_violation_mass_fraction=0.25,
                            warm_repair_budget=2.0, large_repair_mass_fraction=0.125)
    assert s.start == wnet.StartMode.Trash and s.warm == wnet.WarmMode.NONE
    assert s.pivot == wnet.NSPivotRule.FIRST_ELIGIBLE and s.repair_max_nodes == 17
    assert s.warm_violation_limit == 3 and s.warm_violation_mass_fraction == 0.25
    assert s.warm_repair_budget == 2.0 and s.large_repair_mass_fraction == 0.125
    d = wnet.NetworkSimplex()
    assert d.start == wnet.StartMode.Trash and d.warm == wnet.WarmMode.DualRatio
    assert d.repair_max_nodes == (1000 if HAS_CRASH_START else -1)
    with pytest.raises(TypeError):
        wnet.NetworkSimplex(wnet.NSPivotRule.FIRST_ELIGIBLE)   # keyword-only


def test_top_level_exports():
    import wnet
    for name in ("NetworkSimplex", "NSPivotRule", "WarmMode", "StartMode", "CostScaling",
                 "CycleCanceling", "CapacityScaling", "SlopeDP", "ConvexSweep"):
        assert getattr(wnet, name) is getattr(wnet.wnet_cpp, name)


def test_has_crash_start_flag_is_bool():
    assert isinstance(HAS_CRASH_START, bool)
