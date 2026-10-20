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


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("chain", [False, True])
@pytest.mark.parametrize("warm", [WarmMode.NONE, WarmMode.DualRatio, WarmMode.Simple])
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
