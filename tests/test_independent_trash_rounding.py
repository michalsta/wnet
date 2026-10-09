"""Independent-trash shifting must preserve the quantized objective exactly."""

import math

import numpy as np
import pytest

from wnet import Distribution, WassersteinNetwork
from wnet.distances import DistanceMetric
from wnet.wnet_cpp import (
    CapacityScaling,
    CostScaling,
    CycleCanceling,
    NetworkSimplex,
    SlopeDP,
)

# Include both negative and positive cancellation errors and the two NMR scales.
CASES = [
    (0.02, 0.075, 6),
    (0.03, 0.085, 20),
    (0.02, 0.075, 40662879893724),
    (0.03, 0.085, 33591074694816),
]


def rounded(value):
    return math.floor(value + 0.5)


def network(
    exp_cost, theo_cost, scale, distance=0.0, dimension=2, solver=None, chain=False
):
    empirical = Distribution(np.zeros((dimension, 1)), np.array([2.0]))
    positions = np.zeros((dimension, 1))
    positions[0, 0] = distance
    theoretical = Distribution(positions, np.array([3.0]))
    factory = {"split_distance": 1.0} if chain else {"max_distance": 1.0}
    graph = WassersteinNetwork(
        empirical,
        [theoretical],
        DistanceMetric.LINF,
        intensity_scale=1.0,
        round_max_distance=False,
        force_dense_1d=not chain,
        solver=solver,
        **factory,
    )
    graph.set_cost_scaling(scale)
    graph.add_independent_asymmetric_trash(exp_cost, theo_cost)
    graph.build()
    return graph


@pytest.mark.parametrize("exp_cost,theo_cost,scale", CASES)
@pytest.mark.parametrize(
    "solver", [NetworkSimplex, CostScaling, CapacityScaling, CycleCanceling]
)
def test_identical_peaks_have_zero_cost(exp_cost, theo_cost, scale, solver):
    graph = network(exp_cost, theo_cost, scale, solver=solver())
    graph.solve([2.0 / 3.0])
    assert graph.total_cost() == 0.0


@pytest.mark.parametrize("exp_cost,theo_cost,scale", CASES)
@pytest.mark.parametrize("distance", [0.0, 0.025, 0.06, 0.15])
@pytest.mark.parametrize("point", [0.0, 1.0, 2.0])
def test_matches_nonnegative_transport_and_trash_bill(
    exp_cost, theo_cost, scale, distance, point
):
    graph = network(exp_cost, theo_cost, scale, distance=distance)
    graph.solve([point])
    # One peak on each side: either match min(E,T) or trash everything.
    empirical, theoretical = 2, int(3 * point)
    matched = min(empirical, theoretical)
    e_price, t_price = rounded(exp_cost * scale), rounded(theo_cost * scale)
    match_price = rounded(distance * scale)
    bill = min(
        matched * match_price
        + (empirical - matched) * e_price
        + (theoretical - matched) * t_price,
        empirical * e_price + theoretical * t_price,
    )
    assert graph.wnet.total_cost() == bill


@pytest.mark.parametrize("exp_cost,theo_cost,scale", CASES)
@pytest.mark.parametrize("distance", [0.0, 0.025, 0.06, 0.15])
def test_dense_and_chain_costs_agree_at_coarse_scales(
    exp_cost, theo_cost, scale, distance
):
    dense = network(exp_cost, theo_cost, scale, distance, dimension=1)
    chain = network(
        exp_cost, theo_cost, scale, distance, dimension=1, solver=SlopeDP(), chain=True
    )
    for point in [0.0, 1.0, 2.0]:
        dense.solve([point])
        chain.solve([point])
    assert dense.wnet.total_cost() == chain.wnet.total_cost()


@pytest.mark.parametrize(
    "solver", [NetworkSimplex, CostScaling, CapacityScaling, CycleCanceling]
)
@pytest.mark.parametrize(
    "update", ["update_positions_and_solve", "update_positions_and_get_gradient"]
)
@pytest.mark.parametrize("scale", [20, 40662879893724])
def test_position_updates_preserve_independent_trash_shift(solver, update, scale):
    graph = network(0.02, 0.075, scale, solver=solver())
    graph.solve([1.0])
    empirical = Distribution(np.zeros((2, 1)), np.array([2.0]))
    # Include unchanged coordinates and repeated moves, on both sides of the
    # match-versus-trash threshold, with fractional quantized trash prices.
    for distance in [0.0, 0.025, 0.15, 0.06, 0.0]:
        theoretical = Distribution(np.array([[distance], [0.0]]), np.array([3.0]))
        getattr(graph, update)(empirical, [theoretical])
        fresh = network(0.02, 0.075, scale, distance=distance, solver=solver())
        fresh.solve([1.0])
        assert graph.wnet.total_cost() == fresh.wnet.total_cost()
        if solver is NetworkSimplex:
            cut = graph.dual_cut()
            assert (
                cut["intercept"] + cut["gradient"] @ [1.0] <= fresh.total_cost() + 1e-9
            )
            assert cut["upper_bound"] >= fresh.total_cost() - 1e-9
