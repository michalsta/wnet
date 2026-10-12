"""Storage alternatives preserve solver semantics across warm operations."""

import numpy as np
import pytest

from test_convex_sweep import build, d1, gen
from wnet.wnet_cpp import ConvexSweep, SlopeDP


@pytest.mark.parametrize("cls", [SlopeDP, ConvexSweep])
@pytest.mark.parametrize("storage", ["tree", "augmented"])
def test_configuration_validation(cls, storage):
    config = cls()
    assert config.storage == "flat"
    config.storage = storage
    assert config.storage == storage
    with pytest.raises(ValueError, match="storage"):
        config.storage = "unknown"
    assert config.storage == storage
    with pytest.raises(ValueError, match="storage"):
        cls(storage="unknown")
    if cls is SlopeDP:
        config.dual_mode = "cluster"
        assert config.storage == storage
        config.storage = "flat"
        assert config.dual_mode == "cluster"


@pytest.mark.parametrize("p", [1, 1.5, 2, 3])
@pytest.mark.parametrize("independent", [False, True])
@pytest.mark.parametrize("span", [4, 40])
@pytest.mark.parametrize("storage", ["tree", "augmented"])
def test_solve_and_marginal_parity(p, independent, span, storage):
    rng = np.random.default_rng(810)
    for _ in range(12):
        empirical, targets = gen(rng, span)
        kinds = ["sweep"] + (["slope"] if p == 1 else [])
        for kind in kinds:
            flat = build(empirical, targets, kind, independent, p, 2, storage="flat")
            tree = build(empirical, targets, kind, independent, p, 2, storage=storage)
            for point in [
                np.ones(len(targets)),
                rng.uniform(0, 2, len(targets)),
                np.zeros(len(targets)),
            ]:
                flat.solve(point)
                tree.solve(point)
                assert tree.total_cost() == flat.total_cost()
                np.testing.assert_array_equal(
                    tree.spectrum_proportion_derivatives(),
                    flat.spectrum_proportion_derivatives(),
                )


@pytest.mark.parametrize("storage", ["tree", "augmented"])
def test_tree_configuration_survives_precision_rebuild(storage):
    for config in [SlopeDP("cluster", storage=storage), ConvexSweep(storage=storage)]:
        from wnet import WassersteinNetwork
        from wnet.distances import DistanceMetric

        graph = WassersteinNetwork(
            d1([0, 1, 2], [0.4, 0.3, 0.3]),
            [d1([0, 2], [0.5, 0.5])],
            DistanceMetric.L1,
            split_distance=10,
            intensity_scale=13,
            solver=config,
            p=1 if isinstance(config, SlopeDP) else 2,
        )
        graph.add_independent_asymmetric_trash(3, 5)
        graph.build()
        graph.solve([0.7], with_duals=True)
        graph.dual_cut()
        graph.refine_intensity_precision(10)
        assert graph._solver.storage == storage
        graph.solve([0.7], with_duals=True)
        graph.dual_cut()
