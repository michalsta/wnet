"""NetworkSimplex(repair_max_nodes=...) gates warm dual repair per subgraph."""
import numpy as np
import pytest

from wnet import Distribution, WassersteinNetwork
from wnet.distances import DistanceMetric
from wnet.wnet_cpp import NetworkSimplex, WarmMode


def run(repair_max_nodes, warm=WarmMode.DualRatio, seed=5):
    rng = np.random.default_rng(seed)
    p1, p2 = rng.uniform(0, 40, (2, 200)), rng.uniform(0, 40, (2, 200))
    cfg = NetworkSimplex()
    cfg.warm = warm
    cfg.repair_max_nodes = repair_max_nodes
    net = WassersteinNetwork(Distribution(p1, rng.uniform(0.2, 1, 200)),
                             [Distribution(p2, rng.uniform(0.2, 1, 200))],
                             DistanceMetric.L1, max_distance=8, solver=cfg)
    net.add_simple_trash(8)
    net.build()
    costs = []
    for w in np.linspace(0.5, 1.6, 12):
        net.solve([w])
        costs.append(net.total_cost())
    sizes = [net.get_subgraph(i).no_nodes() for i in range(net.no_subgraphs())]
    return np.array(costs), net.wnet.dual_repair_count(), net.wnet.cold_start_count(), max(sizes)


def test_default_is_disabled():
    import wnet.wnet_cpp as c
    # gated by default only where cold starts are crash starts (see NetworkSimplexConfig)
    assert NetworkSimplex().repair_max_nodes == (1000 if getattr(c, "HAS_CRASH_START", False) else -1)


def test_gate_sends_large_subgraphs_cold_without_changing_costs():
    free, repairs_free, cold_free, largest = run(-1)
    assert repairs_free > 0                       # repair is exercised ungated
    gated, repairs_gated, cold_gated, _ = run(largest - 1)
    assert repairs_gated == 0                     # the largest component never repairs
    assert cold_gated > cold_free
    np.testing.assert_allclose(gated, free, rtol=1e-12)


def test_threshold_at_or_above_size_changes_nothing():
    free = run(-1)
    _, _, _, largest = free
    above = run(largest)
    np.testing.assert_array_equal(above[0], free[0])
    assert above[1:3] == free[1:3]
