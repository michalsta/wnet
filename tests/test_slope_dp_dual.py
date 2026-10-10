"""DP recovered duals checked against independent continuous matching LPs."""
import numpy as np
import pytest
from wnet.wnet_cpp import SlopeDP, WarmMode
from test_dual_cut import fixture


@pytest.mark.parametrize('mode', ['lazy', 'cluster'])
@pytest.mark.parametrize('model', ['simple', 'both', 'independent', 'experimental', 'theoretical'])
@pytest.mark.parametrize('scale', [1, 17.3, 100000])
def test_dp_global_bounds(mode, model, scale):
    for seed in range(4):
        graph, objective = fixture(model, scale, WarmMode.NONE, seed=seed,
                                   chain=True, backend=SlopeDP(dual_mode=mode))
        probes = np.vstack([np.zeros(2), np.ones(2), np.random.default_rng(seed).uniform(0, 3, (8, 2))])
        truths = [objective(w) for w in probes]
        with pytest.raises(RuntimeError, match='solve'):
            graph.dual_cut()
        for w in probes[:5]:
            graph.solve(w)
            cut = graph.dual_cut()
            assert objective(w) <= cut['upper_bound'] + 1e-8
            assert abs(objective(w) - graph.total_cost()) <= cut['rounding_error'] + 1e-8
            for point, value in zip(probes, truths):
                assert cut['intercept'] + cut['gradient'] @ point <= value + 1e-8
            if scale == 100000:
                assert objective(w) - (cut['intercept'] + cut['gradient'] @ w) <= 2 * cut['rounding_error'] + 1e-8


@pytest.mark.parametrize('mode', ['lazy', 'cluster'])
def test_recovery_is_optional_cached_and_invalidated(mode):
    graph, _ = fixture('independent', 100000, WarmMode.NONE, chain=True, backend=SlopeDP(mode))
    graph.solve([1, 1])
    sg = graph.get_subgraph(0)
    recoveries, clusters, potentials = sg.slope_dp_dual_stats()
    assert recoveries == potentials == 0
    assert (clusters > 0) == (mode == 'cluster')
    first = graph.dual_cut()
    assert sg.slope_dp_dual_stats()[0] == 1
    second = graph.dual_cut()
    np.testing.assert_array_equal(first['gradient'], second['gradient'])
    assert sg.slope_dp_dual_stats()[0] == 1
    graph.solve([0.2, 0.7])
    graph.dual_cut()
    assert sg.slope_dp_dual_stats()[0] == 2
    graph.refine_intensity_precision(10)
    with pytest.raises(RuntimeError, match='solve'):
        graph.dual_cut()
    graph.solve([0.2, 0.7])
    graph.dual_cut()


def test_modes_validate_configuration():
    cfg = SlopeDP()
    assert cfg.dual_mode == 'lazy'
    cfg.dual_mode = 'cluster'
    assert cfg.dual_mode == 'cluster'
    with pytest.raises(ValueError, match='dual_mode'):
        cfg.dual_mode = 'unknown'
    with pytest.raises(ValueError, match='dual_mode'):
        SlopeDP('unknown')


@pytest.mark.parametrize('mode', ['lazy', 'cluster'])
@pytest.mark.parametrize('integer', [False, True])
def test_position_updates_refresh_potentials_and_handle_zero_reward(mode, integer):
    from wnet import Distribution, WassersteinNetwork
    from wnet.distances import DistanceMetric
    dtype = np.int64 if integer else float
    def dist(xs, vals):
        return Distribution(np.array([xs], float), np.array(vals, dtype=dtype))
    graph = WassersteinNetwork(dist([0, 2], [1, 1]), [dist([1], [2])],
                               DistanceMetric.L1, split_distance=10,
                               intensity_scale=100, solver=SlopeDP(mode))
    graph.add_simple_trash(3)
    graph.build()
    graph.solve([1])
    first = graph.dual_cut()
    assert first['upper_bound'] == pytest.approx(2)
    graph.update_positions_and_solve(dist([0, 2], [1, 1]), [dist([0], [2])])
    second = graph.dual_cut()
    assert second['intercept'] + second['gradient'] @ [1] <= 2 + 1e-8
    assert second['upper_bound'] == pytest.approx(2)
    assert graph.get_subgraph(0).slope_dp_dual_stats()[0] == 2
    graph.solve([0])
    graph.dual_cut()
    # Independent trash with no matching reward exercises the skipped DP path.
    zero = WassersteinNetwork(dist([0], [1]), [dist([0], [1])], DistanceMetric.L1,
                              split_distance=10, solver=SlopeDP(mode))
    zero.add_independent_asymmetric_trash(0, 0)
    zero.build()
    zero.solve([1])
    assert zero.dual_cut()['upper_bound'] == pytest.approx(0)


@pytest.mark.parametrize('mode', ['lazy', 'cluster'])
@pytest.mark.parametrize('model,expected', [('simple', 4), ('independent', 9)])
def test_disconnected_chains_and_isolated_peaks(mode, model, expected):
    from wnet import Distribution, WassersteinNetwork
    from wnet.distances import DistanceMetric
    empirical = Distribution(np.array([[0., 100.]]), np.ones(2))
    target = Distribution(np.array([[1., 200.]]), np.ones(2))
    graph = WassersteinNetwork(empirical, [target], DistanceMetric.L1,
                               split_distance=10, solver=SlopeDP(mode))
    if model == 'simple':
        graph.add_simple_trash(3)
    else:
        graph.add_independent_asymmetric_trash(3, 5)
    graph.build()
    graph.solve([1])
    cut = graph.dual_cut()
    assert cut['upper_bound'] == pytest.approx(expected)
    assert cut['intercept'] + cut['gradient'] @ [1] == pytest.approx(expected)
