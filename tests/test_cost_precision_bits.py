"""Automatic precision targets are independent of arithmetic safety limits."""

import numpy as np
import pytest

from wnet import Distribution, WassersteinNetwork
from wnet.distances import DistanceMetric
from wnet.wnet_cpp import ConvexSweep, NetworkSimplex, SlopeDP


def network(solver, p, bits=None, intensity_scale=100):
    graph = WassersteinNetwork(
        Distribution(np.array([[0.0]]), np.array([1.0])),
        [Distribution(np.array([[1.0]]), np.array([1.0]))],
        DistanceMetric.L1,
        p=p,
        solver=solver,
        intensity_scale=intensity_scale,
        cost_precision_bits=bits,
        force_dense_1d=isinstance(solver, NetworkSimplex),
    )
    graph.add_independent_asymmetric_trash(2, 2)
    return graph


@pytest.mark.parametrize(
    "solver,p,reference",
    [
        (NetworkSimplex(), 2, 4),
        (SlopeDP(), 1, 4),
        (ConvexSweep(), 1, 4),
        (ConvexSweep(), 2, 4),
    ],
)
@pytest.mark.parametrize("bits", [8, 16, 24])
def test_target_sets_scaled_cost_range(solver, p, reference, bits):
    graph = network(solver, p, bits)
    graph.build()
    assert graph.scale_factor() == 2**bits // reference
    graph.solve([1], with_duals=True)
    assert graph.total_cost() == pytest.approx(1)
    assert graph.dual_cut()["rounded_cost"] == pytest.approx(1)


@pytest.mark.parametrize("p", [1.5, 3])
def test_target_for_other_convex_powers(p):
    graph = network(ConvexSweep(), p, 16)
    graph.build()
    assert graph.scale_factor() == 2**16 // 4
    graph.solve([1], with_duals=True)
    error = graph.get_subgraph(0).convex_sweep_cost_quantization_error()
    assert abs(graph.total_cost() - 1) <= error
    assert graph.dual_cut()["rounded_cost"] == pytest.approx(graph.total_cost())


@pytest.mark.parametrize("storage", ["flat", "tree", "augmented"])
def test_target_grid_matches_explicit_model_and_survives_refinement(storage):
    empirical = Distribution(np.array([[0.19, 0.392, 0.577]]), np.ones(3))
    theoretical = Distribution(np.array([[0.577, 0.934, 1.017, 1.831]]), np.ones(4))

    def make():
        graph = WassersteinNetwork(
            empirical,
            [theoretical],
            DistanceMetric.L1,
            p=2,
            solver=ConvexSweep(storage=storage),
            intensity_scale=100,
        )
        graph.add_independent_asymmetric_trash(2, 2)
        return graph

    graph = make()
    graph.set_cost_precision_bits(np.int64(12))
    graph.build()
    explicit = make()
    explicit.set_cost_scaling(1024)
    explicit.build()
    for point in [[0], [0.5], [1]]:
        graph.solve(point, with_duals=True)
        explicit.solve(point, with_duals=True)
        assert graph.total_cost() == explicit.total_cost()
        assert graph.dual_cut()["rounded_cost"] == explicit.dual_cut()["rounded_cost"]
    graph.refine_intensity_precision(10)
    assert graph.scale_factor() == 1024
    graph.solve([1], with_duals=True)
    assert graph.total_cost() == explicit.total_cost()


@pytest.mark.parametrize("setter_first", [True, False])
def test_positive_explicit_scale_overrides_target(setter_first):
    graph = network(ConvexSweep(), 2)
    if setter_first:
        graph.set_cost_precision_bits(8)
        graph.set_cost_scaling(10000)
    else:
        graph.set_cost_scaling(10000)
        graph.set_cost_precision_bits(8)
    graph.build()
    assert graph.scale_factor() == 10000


def test_target_does_not_weaken_accumulator_guard():
    graph = network(ConvexSweep(), 2, 52, intensity_scale=1e15)
    default = network(ConvexSweep(), 2, intensity_scale=1e15)
    graph.build()
    default.build()
    assert graph.scale_factor() == default.scale_factor() < 2**52 // 4
    graph.solve([1])
    with pytest.raises(OverflowError, match="budget"):
        graph.solve([1000])


def test_minimum_scale_and_existing_legacy_default():
    graph = network(ConvexSweep(), 2, 1)
    graph.build()
    assert graph.scale_factor() == 1
    legacy = network(SlopeDP(), 1)
    legacy.build()
    assert legacy.scale_factor() == 1


@pytest.mark.parametrize("bits", [0, -1, 53, True, np.bool_(True), 1.5, "24", None])
def test_invalid_targets_are_rejected_at_setter(bits):
    graph = network(ConvexSweep(), 2)
    with pytest.raises(ValueError, match="cost_precision_bits"):
        graph.set_cost_precision_bits(bits)


def test_post_build_rejection_and_constructor_validation():
    graph = network(ConvexSweep(), 2)
    graph.build()
    with pytest.raises(RuntimeError, match="before build"):
        graph.set_cost_precision_bits(24)
    with pytest.raises(RuntimeError, match="before build"):
        graph.wnet.set_cost_precision_bits(24)
    with pytest.raises(ValueError, match="cost_precision_bits"):
        network(ConvexSweep(), 2, 53)
