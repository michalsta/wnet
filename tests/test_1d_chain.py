"""
Parity tests for the 1D chain-optimized factory.

Compares CWassersteinNetworkFactory.create_1d against the current dense
CWassersteinNetworkFactory.create on a variety of 1D inputs. For each case
total_cost must match exactly (costs are integer-valued when intensities
and positions round to integers, and the 1D optimal transport cost is the
same regardless of graph encoding).
"""

import itertools

import numpy as np
import pytest

from wnet.distribution import Distribution_1D
from wnet.wnet_cpp import CWassersteinNetworkFactory, DistanceMetric


def _run(factory_fn, base, targets, trash_cost, max_dist):
    vec_base = base.vecdist()
    vec_targets = [t.vecdist() for t in targets]
    net = factory_fn(vec_base, vec_targets, DistanceMetric.L1, max_dist)
    if trash_cost is not None:
        net.add_simple_trash(trash_cost)
    net.build()
    net.solve()
    return net.total_cost(), net


def _cost_pair(base, targets, trash_cost, max_dist):
    dense_cost, dense_net = _run(
        CWassersteinNetworkFactory.create, base, targets, trash_cost, max_dist)
    chain_cost, chain_net = _run(
        CWassersteinNetworkFactory.create_1d, base, targets, trash_cost, max_dist)
    return dense_cost, chain_cost, dense_net, chain_net


MAX_VALUE = CWassersteinNetworkFactory.create_1d.__doc__  # ensure bound


@pytest.mark.parametrize("trash_cost", [None, 5, 100])
def test_basic_matched(trash_cost):
    """Two close peaks with equal mass — trivial matching."""
    base = Distribution_1D(np.array([0.0]), np.array([10]))
    target = Distribution_1D(np.array([3.0]), np.array([10]))
    dense, chain, _, _ = _cost_pair(base, [target], trash_cost, 1000)
    assert dense == chain == 30


@pytest.mark.parametrize("trash_cost", [None, 10])
def test_multiple_peaks(trash_cost):
    """Multiple peaks on both sides, overlapping range."""
    base = Distribution_1D(
        np.array([0.0, 5.0, 10.0]), np.array([4, 6, 2]))
    target = Distribution_1D(
        np.array([1.0, 6.0, 9.0]), np.array([4, 6, 2]))
    dense, chain, _, _ = _cost_pair(base, [target], trash_cost, 1000)
    assert dense == chain


def test_unmatched_far_cluster_dropped():
    """
    Empirical cluster far from any theoretical — today's dense factory drops
    the empirical mass entirely (dead-end). The chain pre-pass must do the
    same: single-side run → no chain edges → dead-end → mass dropped.
    """
    # E peaks at 0,1,2 (cluster); T peaks at 100 (single).
    base = Distribution_1D(np.array([0.0, 1.0, 2.0]), np.array([5, 5, 5]))
    target = Distribution_1D(np.array([100.0]), np.array([15]))
    # max_dist small enough that no cross-side matching is possible.
    dense, chain, _, _ = _cost_pair(base, [target], 50, 5)
    assert dense == chain
    # Both should drop all mass (empirical and theoretical), total cost 0.
    assert dense == 0


# Empty-input behavior: both factories reject empty empirical/theoretical
# distributions with an exception (previously the dense factory hit UB
# inside CloserThanIter on empty inputs). Parity means both raise.


@pytest.mark.parametrize(
    "factory_fn",
    [CWassersteinNetworkFactory.create, CWassersteinNetworkFactory.create_1d])
def test_empty_empirical_raises(factory_fn):
    base = Distribution_1D(
        np.array([], dtype=np.float64), np.array([], dtype=np.int64))
    target = Distribution_1D(np.array([1.0, 2.0]), np.array([3, 4]))
    with pytest.raises(ValueError):
        _run(factory_fn, base, [target], 5, 100)


@pytest.mark.parametrize(
    "factory_fn",
    [CWassersteinNetworkFactory.create, CWassersteinNetworkFactory.create_1d])
def test_empty_theoretical_raises(factory_fn):
    base = Distribution_1D(np.array([1.0, 2.0]), np.array([3, 4]))
    target = Distribution_1D(
        np.array([], dtype=np.float64), np.array([], dtype=np.int64))
    with pytest.raises(ValueError):
        _run(factory_fn, base, [target], 5, 100)


def test_coincident_positions():
    """Empirical and theoretical at the same position — zero-cost chain edge."""
    base = Distribution_1D(np.array([5.0, 10.0]), np.array([3, 7]))
    target = Distribution_1D(np.array([5.0, 10.0]), np.array([3, 7]))
    dense, chain, _, _ = _cost_pair(base, [target], None, 100)
    assert dense == chain == 0


def test_multi_spectrum():
    """Two theoretical spectra sharing the chain."""
    base = Distribution_1D(np.array([0.0, 5.0, 10.0]), np.array([4, 6, 2]))
    t1 = Distribution_1D(np.array([1.0, 6.0]), np.array([4, 6]))
    t2 = Distribution_1D(np.array([9.5]), np.array([2]))
    dense, chain, _, _ = _cost_pair(base, [t1, t2], None, 100)
    assert dense == chain


def test_truncation_with_mixed_runs():
    """
    Two clusters: one with both sides (should match), one empirical-only
    (should drop). Verifies pre-pass per-run logic.
    """
    # Cluster A: E at 0,1; T at 2.  Cluster B (far right): E at 100,101.
    base = Distribution_1D(
        np.array([0.0, 1.0, 100.0, 101.0]), np.array([3, 3, 5, 5]))
    target = Distribution_1D(np.array([2.0]), np.array([6]))
    # max_dist = 5 breaks between the two clusters; B has no theoreticals.
    dense, chain, _, _ = _cost_pair(base, [target], 20, 5)
    assert dense == chain


@pytest.mark.parametrize("seed", range(5))
def test_random_parity(seed):
    """Randomized parity on integer-position 1D inputs."""
    rng = np.random.default_rng(seed)
    m = int(rng.integers(1, 30))
    n = int(rng.integers(1, 30))
    # Use integer positions to avoid float rounding differences between
    # chain gap-cost and dense all-pairs distance.
    e_pos = rng.integers(0, 200, size=m).astype(np.float64)
    t_pos = rng.integers(0, 200, size=n).astype(np.float64)
    e_int = rng.integers(1, 10, size=m).astype(np.int64)
    t_int = rng.integers(1, 10, size=n).astype(np.int64)
    base = Distribution_1D(e_pos, e_int)
    target = Distribution_1D(t_pos, t_int)
    trash_cost = 50
    max_dist = int(rng.integers(5, 300))
    dense, chain, _, _ = _cost_pair(base, [target], trash_cost, max_dist)
    assert dense == chain, (
        f"seed={seed} m={m} n={n} max_dist={max_dist} "
        f"dense={dense} chain={chain}")


def test_chain_edge_count():
    """Sanity-check that chain creates O(m+n) edges, not O(m·n)."""
    base = Distribution_1D(
        np.arange(10, dtype=np.float64), np.ones(10, dtype=np.int64))
    target = Distribution_1D(
        np.arange(10, dtype=np.float64) + 0.5, np.ones(10, dtype=np.int64))
    _, _, dense_net, chain_net = _cost_pair(base, [target], None, 100)
    # Chain: 20 nodes in one run → 19 adjacencies × 2 arcs = 38 chain edges.
    assert chain_net.count_chain_edges() == 38
    assert chain_net.count_matching_edges() == 0
    # Dense: pairwise within max_dist — 10×10 = 100 matching edges.
    assert dense_net.count_matching_edges() == 100
    assert dense_net.count_chain_edges() == 0
