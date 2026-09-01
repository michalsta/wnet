"""
Regression tests: the 1D chain factory must participate in automatic cost
scaling (set_cost_scaling) exactly like the dense factory.

Historical bug: create_1d never tallied its gap costs into the network's
max_real_cost, so a *trash-less* chain network kept scale_factor() == 1 under
set_cost_scaling() auto mode and llround()ed fractional gap costs at scale 1
(a 0.6 gap was priced as 1).  Any add_*_trash call masked the bug by feeding
the trash cost into max_real_cost.  WassersteinDistance used to force the
dense factory for fractional p == 1 positions purely because of this.
"""

import numpy as np
import pytest

from wnet import WassersteinNetwork
from wnet.distribution import Distribution_1D
from wnet.distances import DistanceMetric
from wnet.wasserstein import WassersteinDistance


def d1(pos, intens):
    return Distribution_1D(
        np.asarray(pos, dtype=float), np.asarray(intens, dtype=float)
    )


def w1_reference(pos1, i1, pos2, i2):
    """Exact 1-D W1 (unnormalised, equal total masses): integral of the
    absolute CDF difference — independent of any wnet code path."""
    pos1, i1 = np.asarray(pos1, float), np.asarray(i1, float)
    pos2, i2 = np.asarray(pos2, float), np.asarray(i2, float)
    xs = np.unique(np.concatenate([pos1, pos2]))
    cdf1 = np.array([i1[pos1 <= x].sum() for x in xs])
    cdf2 = np.array([i2[pos2 <= x].sum() for x in xs])
    return float(np.sum(np.abs(cdf1 - cdf2)[:-1] * np.diff(xs)))


def test_trashless_chain_picks_auto_cost_scale():
    # Trash-less chain network + auto cost scaling: the gap costs alone must
    # drive the scale choice (used to stay at 1).
    net = WassersteinNetwork(
        d1([0.0, 1.3], [2.0, 3.0]),
        [d1([0.6, 2.0], [4.0, 1.0])],
        DistanceMetric.L1,
    )
    net.set_cost_scaling()
    net.build()
    net.solve([1.0])
    assert net.count_chain_edges() > 0  # chain factory actually in use
    assert net.scale_factor() > 1  # auto scale chosen from the gap costs


def test_legacy_integer_chain_keeps_scale_one():
    # Without set_cost_scaling, p == 1 stays in legacy truncation mode.
    net = WassersteinNetwork(
        d1([0.0, 5.0], [2.0, 3.0]),
        [d1([1.0, 9.0], [4.0, 1.0])],
        DistanceMetric.L1,
    )
    net.build()
    net.solve([1.0])
    assert net.count_chain_edges() > 0
    assert net.scale_factor() == 1


def test_fractional_gap_priced_exactly():
    # The canonical symptom: two unit masses 0.6 apart must cost 0.6, not 1
    # (llround at scale 1) and not 0 (legacy truncation).
    w = WassersteinDistance(d1([0.0], [1.0]), d1([0.6], [1.0]), DistanceMetric.L1)
    assert w == pytest.approx(0.6, rel=1e-12)


def test_fractional_chain_matches_dense_and_reference():
    # WassersteinDistance no longer forces the dense factory for fractional
    # p == 1 positions: chain path, forced dense, and the independent CDF
    # reference must all agree.
    rng = np.random.default_rng(7)
    for _ in range(8):
        n = int(rng.integers(2, 15))
        m = int(rng.integers(2, 15))
        pos1 = np.sort(rng.uniform(0, 50, n))
        pos2 = np.sort(rng.uniform(0, 50, m))
        i1 = rng.integers(1, 20, n).astype(float)
        total = int(i1.sum())
        cuts = np.sort(rng.integers(1, total, m - 1))
        i2 = np.diff(np.concatenate([[0], cuts, [total]])).astype(float)
        if (i2 <= 0).any():
            continue
        D1, D2 = d1(pos1, i1), d1(pos2, i2)
        w_chain = WassersteinDistance(D1, D2, DistanceMetric.L1)
        w_dense = WassersteinDistance(D1, D2, DistanceMetric.L1, force_dense_1d=True)
        ref = w1_reference(pos1, i1, pos2, i2)
        assert w_chain == pytest.approx(w_dense, rel=1e-9)
        assert w_chain == pytest.approx(ref, rel=1e-9)


def test_integer_positions_stay_bit_exact_with_legacy():
    # All-integer inputs must keep the legacy bit-exact path (scale 1,
    # truncation) and equal the exact reference with zero tolerance.
    rng = np.random.default_rng(11)
    for _ in range(8):
        n = int(rng.integers(2, 12))
        m = int(rng.integers(2, 12))
        pos1 = np.sort(rng.choice(200, n, replace=False)).astype(float)
        pos2 = np.sort(rng.choice(200, m, replace=False)).astype(float)
        i1 = rng.integers(1, 30, n).astype(float)
        total = int(i1.sum())
        cuts = np.sort(rng.integers(1, total, m - 1))
        i2 = np.diff(np.concatenate([[0], cuts, [total]])).astype(float)
        if (i2 <= 0).any():
            continue
        D1, D2 = d1(pos1, i1), d1(pos2, i2)
        w_chain = WassersteinDistance(D1, D2, DistanceMetric.L1)
        w_dense = WassersteinDistance(D1, D2, DistanceMetric.L1, force_dense_1d=True)
        ref = w1_reference(pos1, i1, pos2, i2)
        assert w_chain == w_dense
        assert w_chain == ref
