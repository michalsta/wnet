"""Supply refinement must preserve the moved graph, rather than reconstruct it."""

import numpy as np
import pytest

from wnet import Distribution, WassersteinNetwork
from wnet.distances import DistanceMetric
from wnet.wnet_cpp import (
    CapacityScaling,
    ConvexSweep,
    CostScaling,
    CycleCanceling,
    NetworkSimplex,
    SlopeDP,
    WarmMode,
)


def simplex(mode):
    solver = NetworkSimplex()
    solver.warm = mode
    return solver


@pytest.mark.parametrize(
    "solver,chain,p",
    [
        (simplex(WarmMode.NONE), False, 1),
        (simplex(WarmMode.Dual), False, 1),
        (simplex(WarmMode.LinkCut), False, 1),
        (CostScaling(), False, 1),
        (CapacityScaling(), False, 1),
        (CycleCanceling(), False, 1),
        (simplex(WarmMode.Dual), True, 1),
        (SlopeDP(), True, 1),
        (ConvexSweep(), True, 2),
    ],
)
@pytest.mark.parametrize(
    "update", ["update_positions_and_solve", "update_positions_and_get_gradient"]
)
def test_refinement_retains_moved_costs_and_fixed_topology(solver, chain, p, update):
    if isinstance(solver, ConvexSweep) and update.endswith("gradient"):
        pytest.skip("ConvexSweep does not expose position gradients")

    def distribution(x):
        positions = [[x]] if chain else [[x], [0.0]]
        return Distribution(np.array(positions, dtype=float), np.array([1.0]))

    base, target = distribution(0.0), distribution(1.0)
    kwargs = {"split_distance": 1.5} if chain else {"max_distance": 1.5}
    graph = WassersteinNetwork(
        base,
        [target],
        DistanceMetric.L1,
        solver=solver,
        p=p,
        intensity_scale=1,
        round_max_distance=False,
        **kwargs,
    )
    graph.set_cost_scaling(100)
    graph.add_simple_trash(10)
    graph.build()
    graph.solve([1.0])
    assert graph.total_cost() == pytest.approx(1)

    # The retained edge now crosses the original cap/split radius. Recreating
    # the factory from either the original or moved distributions is wrong.
    getattr(graph, update)(base, [distribution(2.0)])
    assert graph.total_cost() == pytest.approx(2**p)
    topology = (
        graph.no_subgraphs(),
        graph.count_matching_edges(),
        graph.count_chain_edges(),
    )
    old_network = graph.wnet
    graph.refine_intensity_precision(10)
    assert graph.wnet is not old_network
    assert graph.intensity_scale_factor() == 10
    assert (
        graph.no_subgraphs(),
        graph.count_matching_edges(),
        graph.count_chain_edges(),
    ) == topology
    with pytest.raises(RuntimeError, match="solve"):
        graph.total_cost()
    graph.solve([1.0])
    assert graph.total_cost() == pytest.approx(2**p)
    # An externally retained reference to the old network is still valid.
    assert old_network.total_cost() / (
        old_network.scale_factor() * old_network.intensity_scale_factor()
    ) == pytest.approx(2**p)

    # Repeated refinement and updates must keep using the current geometry.
    getattr(graph, update)(base, [distribution(3.0)])
    graph.refine_intensity_precision(10)
    graph.solve([1.0])
    assert graph.total_cost() == pytest.approx(3**p)


def test_failed_refinement_preserves_solution_and_settings(monkeypatch):
    base = Distribution(np.array([[0.0], [0.0]]), np.array([1.0]))
    graph = WassersteinNetwork(base, [base], DistanceMetric.L1, intensity_scale=1)
    graph.set_cost_scaling(100)
    graph.add_simple_trash(10)
    graph.build()
    graph.solve([1.0])
    original = graph.wnet

    def fail(*args):
        raise OverflowError("rebuild failed")

    monkeypatch.setattr(type(original), "refined_copy", fail)
    with pytest.raises(OverflowError, match="rebuild failed"):
        graph.refine_intensity_precision(10)
    assert graph.wnet is original
    assert graph._intensity_scale_arg == 1
    assert graph._cost_scaling_arg == 100
    assert graph.intensity_scale_factor() == 1
    assert graph.total_cost() == 0
    assert graph.dual_cut()["upper_bound"] >= 0


@pytest.mark.parametrize(
    "model", ["simple", "both", "independent", "experimental", "theoretical"]
)
def test_refinement_preserves_trash_models_and_isolated_peaks(model):
    def distribution(positions, intensities):
        return Distribution(np.array([positions, [0.0, 0.0]]), np.array(intensities))

    base = distribution([0.0, 10.0], [1.0, 2.0])
    target = distribution([1.0, 20.0], [1.0, 3.0])
    graph = WassersteinNetwork(
        base,
        [target],
        DistanceMetric.L1,
        max_distance=1.5,
        intensity_scale=1,
        round_max_distance=False,
    )
    if model == "simple":
        graph.add_simple_trash(10)
    elif model == "independent":
        graph.add_independent_asymmetric_trash(3, 5)
    else:
        if model in ("both", "experimental"):
            graph.add_experimental_trash(3)
        if model in ("both", "theoretical"):
            graph.add_theoretical_trash(5)
    graph.build()
    graph.solve([1.0])
    graph.update_positions_and_solve(base, [distribution([2.0, 10.0], [1.0, 3.0])])
    # The isolated peaks have moved into matching range, but refinement must
    # keep them isolated and preserve the original trash model's bill.
    before = graph.total_cost()
    cut_before = graph.dual_cut()
    graph.refine_intensity_precision(10)
    graph.solve([1.0])
    assert graph.no_subgraphs() == 1
    assert graph.count_matching_edges() == 1
    assert graph.total_cost() == before
    cut_after = graph.dual_cut()
    assert cut_after["gradient"] == pytest.approx(cut_before["gradient"])
    assert cut_after["intercept"] == pytest.approx(cut_before["intercept"])
