"""In-place intensity rescaling must reproduce refined_copy() exactly.

refine_intensity_precision() now re-quantises the built network in place
(rescale_intensity) instead of copying and rebuilding it.  Both routes change
only the supply grid and the cost scale, so every observable -- scales, cost,
cut, flows -- has to agree bit for bit, for every backend and trash model,
and the in-place route must be the one taken when no positions have moved.
"""

import numpy as np
import pytest

from wnet import Distribution, WassersteinNetwork
from wnet.distances import DistanceMetric
from wnet.wnet_cpp import ConvexSweep, NetworkSimplex, SlopeDP, WarmMode


def simplex(mode):
    solver = NetworkSimplex()
    solver.warm = mode
    return solver


def make_network(solver, chain, p, trash, rng):
    dim = 1 if chain else 2
    def dist(n):
        pos = np.sort(rng.uniform(0, 10, size=(dim, n)), axis=1)
        return Distribution(pos, rng.uniform(0.2, 2.0, size=n))
    base, targets = dist(14), [dist(9), dist(11)]
    kwargs = {"split_distance": 3.0} if chain else {"max_distance": 3.0}
    graph = WassersteinNetwork(base, targets, DistanceMetric.L1, solver=solver, p=p,
                               intensity_scale=1000.0, round_max_distance=False, **kwargs)
    graph.set_cost_scaling(0)
    if trash == "simple":
        graph.add_simple_trash(1.5 ** p)
    elif trash == "asymmetric":
        graph.add_experimental_trash(1.0 ** p)
        graph.add_theoretical_trash(1.5 ** p)
    else:
        graph.add_independent_asymmetric_trash(0.8 ** p, 0.9 ** p)
    graph.build()
    return graph


def observe(net, point, with_duals, n_targets):
    net.solve(point, with_duals)
    cut = net.dual_cut()
    try:
        flows = [net.flows_for_target(k) for k in range(n_targets)]
    except RuntimeError:  # ConvexSweep reports no per-arc flows
        flows = []
    return (net.scale_factor(), net.intensity_scale_factor(), net.total_cost(),
            {k: (float(v) if np.isscalar(v) else np.asarray(v)) for k, v in cut.items()}, flows)


def assert_same(a, b):
    assert a[0] == b[0] and a[1] == b[1] and a[2] == b[2]
    assert a[3].keys() == b[3].keys()
    for k in a[3]:
        np.testing.assert_array_equal(a[3][k], b[3][k], err_msg=k)
    for fa, fb in zip(a[4], b[4]):
        for xa, xb in zip(fa, fb):
            np.testing.assert_array_equal(xa, xb)


CASES = [
    (simplex(WarmMode.NONE), False, 1, "simple"),
    (simplex(WarmMode.Dual), False, 1, "asymmetric"),
    (simplex(WarmMode.Dual), False, 1, "independent"),
    (simplex(WarmMode.Dual), False, 2, "simple"),
    (simplex(WarmMode.Dual), True, 1, "asymmetric"),
    (SlopeDP(), True, 1, "simple"),
    (SlopeDP(), True, 1, "independent"),
    (ConvexSweep(), True, 2, "independent"),
    (ConvexSweep(), True, 2, "simple"),
]


@pytest.mark.parametrize("solver,chain,p,trash", CASES)
def test_rescale_matches_refined_copy(solver, chain, p, trash):
    rng = np.random.default_rng(3)
    graph = make_network(solver, chain, p, trash, rng)
    sweep = isinstance(solver, ConvexSweep)
    point = [0.6, 0.35]
    graph.solve(point, sweep)
    before = graph.total_cost()

    reference = graph.wnet.refined_copy(graph.intensity_scale_factor() * 10)
    old = graph.wnet
    graph.refine_intensity_precision(10)
    assert graph.wnet is old, "no position moved, so the rescale must be in place"
    assert graph.intensity_scale_factor() == pytest.approx(10_000.0)

    assert_same(observe(graph.wnet, point, sweep, 2), observe(reference, point, sweep, 2))
    assert graph.total_cost() == pytest.approx(before, rel=1e-2)

    # A second step and a different point keep agreeing (warm paths differ).
    reference2 = graph.wnet.refined_copy(graph.intensity_scale_factor() * 10)
    graph.refine_intensity_precision(10)
    assert_same(observe(graph.wnet, [0.2, 0.9], sweep, 2), observe(reference2, [0.2, 0.9], sweep, 2))


def test_rescale_discards_solution_and_can_go_back():
    rng = np.random.default_rng(5)
    graph = make_network(simplex(WarmMode.Dual), False, 1, "simple", rng)
    graph.solve([0.5, 0.5])
    scale0 = graph.intensity_scale_factor()
    cost0 = graph.total_cost()
    graph.wnet.rescale_intensity(scale0 * 100)
    with pytest.raises(RuntimeError):
        graph.wnet.total_cost()
    graph.wnet.rescale_intensity(scale0)
    graph.solve([0.5, 0.5])
    assert graph.total_cost() == cost0


def test_rescale_refuses_after_position_update_and_wrapper_falls_back():
    rng = np.random.default_rng(9)
    graph = make_network(simplex(WarmMode.Dual), False, 1, "simple", rng)
    graph.solve([0.5, 0.5])
    base = graph._base_distribution
    moved = Distribution(base.positions * 1.01, base.intensities)
    graph.update_positions_and_solve(moved, list(graph._target_distributions))
    assert not graph.wnet.can_rescale_in_place()
    with pytest.raises(RuntimeError):
        graph.wnet.rescale_intensity(graph.intensity_scale_factor() * 10)
    reference = graph.wnet.refined_copy(graph.intensity_scale_factor() * 10)
    old = graph.wnet
    graph.refine_intensity_precision(10)
    assert graph.wnet is not old
    assert_same(observe(graph.wnet, [0.5, 0.5], False, 2), observe(reference, [0.5, 0.5], False, 2))


def test_rescale_rejects_bad_scale_without_touching_network():
    rng = np.random.default_rng(11)
    graph = make_network(SlopeDP(), True, 1, "simple", rng)
    graph.solve([0.5, 0.5])
    cost = graph.total_cost()
    with pytest.raises(OverflowError):
        graph.set_intensity_precision(1e300)
    with pytest.raises(OverflowError):
        graph.set_intensity_precision(0.0)
    assert graph.total_cost() == cost
