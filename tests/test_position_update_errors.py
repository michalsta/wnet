"""Rejected inputs are harmless; failed mutations require rebuilding."""

import subprocess
import sys

import numpy as np
import pytest

from wnet import Distribution, WassersteinNetwork
from wnet.distances import DistanceMetric


def distribution(xs, dimension=2):
    positions = np.zeros((dimension, len(xs)))
    positions[0] = xs
    return Distribution(positions, np.ones(len(xs)))


def dense_network():
    base, target = distribution([0, 10]), distribution([1, 11])
    graph = WassersteinNetwork(
        base, [target], DistanceMetric.L1, max_distance=2, intensity_scale=1
    )
    graph.add_simple_trash(10)
    graph.build()
    graph.solve([1])
    return graph, base, target


@pytest.mark.parametrize(
    "update", ["update_positions_and_solve", "update_positions_and_get_gradient"]
)
def test_missing_targets_raise_without_crashing_python(update):
    # A subprocess makes this a regression for the old segmentation fault
    # without allowing it to take the entire pytest process down.
    code = f"""
import sys
if sys.platform != "win32":
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
import numpy as np
from wnet import Distribution, WassersteinNetwork
from wnet.distances import DistanceMetric
base = Distribution(np.array([[0., 10.], [0., 0.]]), np.ones(2))
target = Distribution(np.array([[1., 11.], [0., 0.]]), np.ones(2))
graph = WassersteinNetwork(base, [target], DistanceMetric.L1, max_distance=2, intensity_scale=1)
graph.add_simple_trash(10)
graph.build()
graph.solve([1])
try:
    graph.{update}(base, [])
except ValueError as exc:
    assert 'target count' in str(exc)
else:
    raise AssertionError('missing targets were accepted')
assert graph.total_cost() == 2
graph.solve([1])
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr or f"child exit: {result.returncode}"


@pytest.mark.parametrize(
    "update", ["update_positions_and_solve", "update_positions_and_get_gradient"]
)
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize(
    "invalid", ["extra_target", "empirical_count", "target_count", "dimension"]
)
def test_invalid_shapes_preserve_the_solution(update, native, invalid):
    graph, base, target = dense_network()
    targets = [target]
    if invalid == "extra_target":
        targets.append(target)
        message = "target count"
    elif invalid == "empirical_count":
        base = distribution([0, 10, 20])
        message = "empirical peak count"
    elif invalid == "target_count":
        targets = [distribution([2])]
        message = "target peak count"
    else:
        base = distribution([0, 10], dimension=1)
        targets = [distribution([1, 11], dimension=1)]
        message = "dimension"
    with pytest.raises(ValueError, match=message):
        if native:
            getattr(graph.wnet, update)(
                base.vecdist, [t.vecdist for t in targets], DistanceMetric.L1
            )
        else:
            getattr(graph, update)(base, targets)
    assert graph.total_cost() == 2
    graph.solve([1])
    assert graph.dual_cut()["upper_bound"] >= 2


def chain_network():
    base = distribution([0, 10, 12], dimension=1)
    target = distribution([1, 11, 13], dimension=1)
    graph = WassersteinNetwork(
        base,
        [target],
        DistanceMetric.L1,
        split_distance=1.5,
        round_max_distance=False,
        intensity_scale=1,
    )
    graph.add_simple_trash(10)
    graph.build()
    graph.solve([1])
    assert graph.no_subgraphs() == 2
    return graph, base, target


@pytest.mark.parametrize(
    "update", ["update_positions_and_solve", "update_positions_and_get_gradient"]
)
def test_later_component_failure_invalidates_all_solution_access(update):
    graph, base, target = chain_network()
    retained = [graph.get_subgraph(k) for k in range(2)]
    # Populate derivative caches before invalidation, too.
    graph.signal_part_derivatives()
    graph.spectrum_proportion_derivatives_fast_approx()
    with pytest.raises(
        RuntimeError, match="peaks have crossed.*network state.*[Rr]ebuild"
    ):
        getattr(graph, update)(base, [distribution([2, 9, 13], dimension=1)])

    queries = [
        lambda: graph.solve([1]),
        graph.total_cost,
        graph.dual_cut,
        graph.signal_part_derivatives,
        graph.signal_part_derivatives_fast_approx,
        graph.spectrum_proportion_derivatives,
        graph.spectrum_proportion_derivatives_fast_approx,
        lambda: graph.flows_for_target(0),
        lambda: graph.get_subgraph(0),
        lambda: graph.refine_intensity_precision(10),
        lambda: graph.update_positions_and_solve(base, [target]),
        lambda: graph.update_positions_and_get_gradient(base, [target]),
    ]
    for subgraph in retained:
        assert not subgraph.is_solved()
        queries.extend(
            [
                subgraph.total_cost,
                subgraph.get_flow_map,
                subgraph.dual_values,
                subgraph.signal_part_derivatives,
                subgraph.spectrum_proportion_derivatives_fast_approx,
                lambda sg=subgraph: sg.set_point([1]),
                lambda sg=subgraph: sg.build(),
            ]
        )
    for query in queries:
        with pytest.raises(RuntimeError, match="[Rr]ebuild before reuse"):
            query()

    graph.build()
    graph.solve([1])
    assert graph.total_cost() == 3
    # Rebuilding replaces the network; retained old subgraphs remain invalid.
    for subgraph in retained:
        with pytest.raises(RuntimeError, match="[Rr]ebuild before reuse"):
            subgraph.total_cost()


@pytest.mark.parametrize(
    "update", ["update_positions_and_solve", "update_positions_and_get_gradient"]
)
def test_failure_before_first_mutation_preserves_the_solution(update):
    # Cross peaks in the first component, before any costs are changed.
    base = distribution([0, 2, 10], dimension=1)
    target = distribution([1, 3, 11], dimension=1)
    graph = WassersteinNetwork(
        base,
        [target],
        DistanceMetric.L1,
        split_distance=1.5,
        round_max_distance=False,
        intensity_scale=1,
    )
    graph.add_simple_trash(10)
    graph.build()
    graph.solve([1])
    with pytest.raises(ValueError, match="peaks have crossed"):
        getattr(graph, update)(base, [distribution([-1, 3, 11], dimension=1)])
    assert graph.total_cost() == 3
    graph.solve([1])


@pytest.mark.parametrize(
    "update", ["update_positions_and_solve", "update_positions_and_get_gradient"]
)
@pytest.mark.parametrize("later", [False, True])
def test_cost_overflow_invalidates_only_after_mutation_starts(update, later):
    graph, base, _ = dense_network()
    moved = distribution([2, 1e30] if later else [1e30, 11])
    if later:
        with pytest.raises(
            RuntimeError, match="overflows.*network state.*Rebuild before reuse"
        ):
            getattr(graph, update)(base, [moved])
        with pytest.raises(RuntimeError, match="Rebuild before reuse"):
            graph.total_cost()
    else:
        with pytest.raises(OverflowError, match="overflows"):
            getattr(graph, update)(base, [moved])
        assert graph.total_cost() == 2
        graph.solve([1])
