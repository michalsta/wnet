"""
Tests for set_flow_budget() and the solve()-time cost-accumulator overflow
guard on WassersteinNetwork.

The auto cost scale is sized at build() from the supplies as constructed
(point == 1); solve(point) multiplies theoretical supplies by the point, so a
large point can push the accumulated integer cost past int64 and wrap the
total negative.  set_flow_budget() declares the expected point-scaled flow so
build() sizes the scale for it, and solve() rejects any point whose
worst-case accumulated cost exceeds the 2^62 budget.
"""

import numpy as np
import pytest

from wnet import WassersteinNetwork
from wnet.distribution import Distribution_1D
from wnet.distances import DistanceMetric


def d1(pos, intens):
    return Distribution_1D(np.array(pos, dtype=float), np.array(intens, dtype=float))


def make_network(budget=None, trash=100.0, intensity_scale=5.0):
    emp = d1([1.0, 100.0], [50.0, 150.0])
    t1 = d1([1.0], [10.0])
    t2 = d1([100.0], [15.0])
    W = WassersteinNetwork(
        emp,
        [t1, t2],
        DistanceMetric.LINF,
        10.0,
        round_max_distance=False,
        intensity_scale=intensity_scale,
    )
    W.add_simple_trash(trash)
    W.set_cost_scaling(0)
    if budget is not None:
        W.set_flow_budget(budget)
    W.build()
    return W


def test_point_one_unchanged_without_budget():
    W = make_network()
    W.solve([1.0, 1.0])
    # emp 200 total, theo 25 at point 1 -> 175 units trashed at cost 100.
    assert W.total_cost() == pytest.approx(17500.0, rel=1e-6)


def test_guard_rejects_points_past_sizing_without_budget():
    # Without a declared budget the auto scale is sized for 4x the build-time
    # supplies (2 bits of headroom over point == 1), so moderate points fit,
    # but the worst-case accumulated cost of points past the 4x sizing exceeds
    # the 2^62 ceiling and solve() must refuse rather than risk a silent int64
    # wrap -- even for points (like [100, 10] here) whose actual cost would
    # have been fine.  Callers that need such points declare them via
    # set_flow_budget().
    W = make_network()
    # Build-time supplies: emp 200 + theo 25 = 225; auto sizing covers 900.
    with pytest.raises(OverflowError, match="set_flow_budget"):
        W.solve([196.0, 0.44])  # point-scaled flow 2166.6 > 900
    with pytest.raises(OverflowError, match="set_flow_budget"):
        W.solve([100.0, 10.0])  # point-scaled flow 1350 > 900


def test_default_headroom_accepts_moderate_points():
    # The 2-bit default headroom exists exactly so that nearby points -- an
    # optimizer probing past 1.0 -- don't trip the guard on a freshly built
    # network.  [2.0, 2.0] (flow 250) and [5.0, 10.0] (flow 400) fit the 4x
    # sizing (900) and must solve; [5, 10] matches all theoretical mass at
    # distance 0, so its cost is 0.
    W = make_network()
    W.solve([2.0, 2.0])
    W.solve([5.0, 10.0])
    assert W.total_cost() == pytest.approx(0.0, abs=1e-6)


def test_budget_widens_sizing_and_point_computes_correctly():
    W = make_network(budget=6600.0)
    unbudgeted_scale = make_network().scale_factor()
    assert W.scale_factor() < unbudgeted_scale
    W.solve([196.0, 0.44])
    # Nothing is transported (both pairs are coincident): 50 units match at
    # t1 and 6.6 at t2.  Simple trash is annihilating and priced network-wide,
    # so the bill is max(emp 200, theo 1966.6) - 56.6 matched = 1910 escaping
    # units at cost 100.
    assert W.total_cost() == pytest.approx(191000.0, rel=1e-6)
    W.solve([5.0, 10.0])
    assert W.total_cost() == pytest.approx(0.0, abs=1e-6)


def test_guard_rejects_points_beyond_declared_budget():
    W = make_network(budget=6600.0)
    with pytest.raises(OverflowError):
        W.solve([1e6, 1e6])


def test_flow_budget_must_precede_build():
    W = make_network()
    with pytest.raises(RuntimeError, match="before build"):
        W.set_flow_budget(1000.0)


def test_flow_budget_validates_input():
    emp = d1([1.0], [10.0])
    t1 = d1([1.0], [2.0])
    W = WassersteinNetwork(
        emp, [t1], DistanceMetric.LINF, 10.0, round_max_distance=False
    )
    with pytest.raises(ValueError):
        W.set_flow_budget(-1.0)
    with pytest.raises(ValueError):
        W.set_flow_budget(float("nan"))
