"""
API coverage tests for distribution.py, wasserstein.py, wasserstein_network.py.

Focuses on the public surface that was previously uncovered: Distribution
validation / utilities, WassersteinDistance(), and WassersteinNetwork
convenience methods.
"""

import pickle
import warnings

import numpy as np
import pytest

from wnet import Distribution, WassersteinNetwork
from wnet.distribution import Distribution_1D
from wnet.distances import DistanceMetric
from wnet.wasserstein import TruncatedWassersteinDistance, WassersteinDistance

try:
    import networkx  # noqa: F401
    HAS_NX = True
except ImportError:
    HAS_NX = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _d1(pos, intens):
    return Distribution_1D(np.array(pos, dtype=float), np.array(intens, dtype=float))


# ---------------------------------------------------------------------------
# Distribution construction validation
# ---------------------------------------------------------------------------

def test_distribution_rejects_1d_positions():
    with pytest.raises(ValueError, match="2D array"):
        Distribution(np.array([1.0, 2.0]), np.array([1.0, 1.0]))


def test_distribution_rejects_intensity_shape_mismatch():
    with pytest.raises(ValueError, match="n_points"):
        Distribution(np.array([[1.0, 2.0]]), np.array([1.0, 1.0, 1.0]))


def test_distribution_rejects_low_dimension():
    with pytest.raises(ValueError, match="dimension"):
        Distribution(np.zeros((0, 3)), np.array([1.0, 1.0, 1.0]))


def test_distribution_rejects_high_dimension():
    with pytest.raises(ValueError, match="dimension"):
        Distribution(np.zeros((21, 1)), np.array([1.0]))


# ---------------------------------------------------------------------------
# Distribution_1D validation
# ---------------------------------------------------------------------------

def test_distribution_1d_rejects_2d_positions():
    with pytest.raises(ValueError, match="1D"):
        Distribution_1D(np.array([[1.0, 2.0]]), np.array([1.0, 1.0]))


def test_distribution_1d_rejects_2d_intensities():
    with pytest.raises(ValueError, match="1D"):
        Distribution_1D(np.array([1.0, 2.0]), np.array([[1.0, 1.0]]))


def test_distribution_1d_rejects_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        Distribution_1D(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0]))


def test_distribution_1d_accepts_lists():
    d = Distribution_1D([1.0, 2.0], [3.0, 4.0])
    assert len(d) == 2


# ---------------------------------------------------------------------------
# Distribution utility methods
# ---------------------------------------------------------------------------

def test_n_highest():
    d = _d1([1.0, 2.0, 3.0], [10.0, 30.0, 20.0])
    top2 = d.n_highest(2)
    assert len(top2) == 2
    assert top2.sum_intensities == pytest.approx(50.0)


def test_p_trim():
    d = _d1([1.0, 2.0, 3.0, 4.0], [10.0, 40.0, 30.0, 20.0])
    trimmed = d.p_trim(0.7)
    # 40+30 = 70% of 100 → keeps the top 2 peaks
    assert len(trimmed) == 2


def test_scaled():
    d = _d1([1.0, 2.0], [3.0, 4.0])
    s = d.scaled(2.0)
    # scaled() scales intensities only, not positions
    assert np.allclose(s.positions, d.positions)
    assert np.allclose(s.intensities, d.intensities * 2.0)


def test_positions_intensities_scaled():
    d = _d1([1.0, 2.0], [3.0, 4.0])
    s = d.positions_intensities_scaled(3.0)
    assert np.allclose(s.positions, d.positions * 3.0)
    assert np.allclose(s.intensities, d.intensities * 3.0)


def test_normalized():
    d = _d1([1.0, 2.0], [3.0, 7.0])
    n = d.normalized()
    assert n.sum_intensities == pytest.approx(1.0)
    assert np.allclose(n.intensities, [0.3, 0.7])


def test_normalized_zero_raises():
    d = _d1([1.0, 2.0], [0.0, 0.0])
    with pytest.raises(ValueError, match="zero total intensity"):
        d.normalized()


def test_as_distribution_returns_base():
    d = _d1([1.0, 2.0], [1.0, 1.0])
    b = d.as_distribution()
    assert type(b) is Distribution
    assert np.allclose(b.positions, d.positions)


def test_bounding_box():
    d = Distribution(np.array([[1.0, 3.0], [2.0, 4.0]]), np.array([1.0, 1.0]))
    lo, hi = d.bounding_box()
    assert np.allclose(lo, [1.0, 2.0])
    assert np.allclose(hi, [3.0, 4.0])


def test_cpp_repr_contains_positions():
    d = _d1([1.0, 2.0], [1.0, 1.0])
    r = d.cpp_repr()
    assert "1.0" in r and "2.0" in r


def test_str_contains_dimension():
    d = _d1([1.0], [1.0])
    assert "dimension=1" in str(d)


# ---------------------------------------------------------------------------
# Distribution pickle round-trip
# ---------------------------------------------------------------------------

def test_distribution_pickle_roundtrip():
    d = _d1([1.0, 2.0, 3.0], [5.0, 10.0, 15.0])
    d2 = pickle.loads(pickle.dumps(d))
    assert np.allclose(d2.positions, d.positions)
    assert np.allclose(d2.intensities, d.intensities)


# ---------------------------------------------------------------------------
# WassersteinDistance (high-level function)
# ---------------------------------------------------------------------------

def test_wasserstein_distance_matched():
    d1 = _d1([0.0], [5.0])
    d2 = _d1([3.0], [5.0])
    # W_1 = sum(d * flow) = 3.0 * 5.0 = 15.0 (unnormalized distributions)
    dist = WassersteinDistance(d1, d2, DistanceMetric.L1)
    assert dist == pytest.approx(15.0)


def test_wasserstein_distance_p2():
    d1 = _d1([0.0], [1.0])
    d2 = _d1([3.0], [1.0])
    # W_2^2 = 3^2 * 1 = 9  →  W_2 = 3.0
    dist = WassersteinDistance(d1, d2, DistanceMetric.L2, p=2)
    assert dist == pytest.approx(3.0, rel=1e-4)


def test_wasserstein_distance_unequal_intensities_raises():
    d1 = _d1([0.0], [1.0])
    d2 = _d1([0.0], [2.0])
    with pytest.raises(RuntimeError, match="same total intensity"):
        WassersteinDistance(d1, d2, DistanceMetric.L1)


# ---------------------------------------------------------------------------
# TruncatedWassersteinDistance
# ---------------------------------------------------------------------------

def test_truncated_wasserstein_distance_basic():
    d1 = _d1([0.0], [5.0])
    d2 = _d1([2.0], [5.0])
    # W_1 = 2.0 * 5.0 = 10.0 (unnormalized, one pair at distance 2, mass 5)
    dist = TruncatedWassersteinDistance(d1, d2, DistanceMetric.L1, max_distance=10)
    assert dist == pytest.approx(10.0)


def test_truncated_wasserstein_unequal_intensities_raises():
    d1 = _d1([0.0], [1.0])
    d2 = _d1([0.0], [3.0])
    with pytest.raises(RuntimeError, match="same total intensity"):
        TruncatedWassersteinDistance(d1, d2, DistanceMetric.L1, max_distance=10)


# ---------------------------------------------------------------------------
# WassersteinNetwork construction guards
# ---------------------------------------------------------------------------

def test_wasserstein_network_str():
    d1 = _d1([0.0], [1.0])
    d2 = _d1([1.0], [1.0])
    W = WassersteinNetwork(d1, [d2], DistanceMetric.L1, max_distance=5)
    W.add_simple_trash(5)
    W.build()
    s = str(W)
    assert isinstance(s, str) and len(s) > 0


def test_unknown_method_raises():
    d1 = _d1([0.0], [1.0])
    d2 = _d1([1.0], [1.0])
    with pytest.raises(ValueError, match="Unknown method"):
        WassersteinNetwork(d1, [d2], DistanceMetric.L1, max_distance=10,
                           method="bogus_algo")


def test_fractional_max_distance_warns():
    d1 = _d1([0.0], [1.0])
    d2 = _d1([1.0], [1.0])
    with pytest.warns(UserWarning, match="not an integer"):
        W = WassersteinNetwork(d1, [d2], DistanceMetric.L1, max_distance=1.7)
    W.add_simple_trash(2)
    W.build()
    W.solve()
    assert W.total_cost() >= 0


# ---------------------------------------------------------------------------
# WassersteinNetwork subgraph methods
# ---------------------------------------------------------------------------

def _solved_network():
    d1 = _d1([0.0, 5.0], [3.0, 2.0])
    d2 = _d1([1.0, 6.0], [3.0, 2.0])
    W = WassersteinNetwork(d1, [d2], DistanceMetric.L1, max_distance=10)
    W.add_simple_trash(10)
    W.build()
    W.solve()
    return W


def test_signal_part_derivatives_fast_approx_shape():
    W = _solved_network()
    for sg in W.subgraphs():
        d = sg.signal_part_derivatives_fast_approx()
        assert isinstance(d, dict)


def test_spectrum_proportion_derivatives_fast_approx_shape():
    W = _solved_network()
    for sg in W.subgraphs():
        arr = sg.spectrum_proportion_derivatives_fast_approx()
        assert isinstance(arr, np.ndarray)


def test_residual_graph_before_solve_raises():
    d1 = _d1([0.0], [1.0])
    d2 = _d1([1.0], [1.0])
    W = WassersteinNetwork(d1, [d2], DistanceMetric.L1, max_distance=5)
    W.add_simple_trash(5)
    W.build()
    for sg in W.subgraphs():
        with pytest.raises(RuntimeError, match="solve()"):
            sg.residual_graph()


@pytest.mark.skipif(not HAS_NX, reason="networkx not installed")
def test_as_networkx_after_solve():
    W = _solved_network()
    for sg in W.subgraphs():
        G = sg.as_networkx()
        assert G.number_of_nodes() > 0
        assert G.number_of_edges() > 0
        for _, _, data in G.edges(data=True):
            assert "flow" in data


@pytest.mark.skipif(not HAS_NX, reason="networkx not installed")
def test_residual_graph_after_solve():
    W = _solved_network()
    for sg in W.subgraphs():
        R = sg.residual_graph()
        assert R.number_of_nodes() > 0
