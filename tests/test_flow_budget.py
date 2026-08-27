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
    # Without a declared budget the auto scale is sized for the build-time
    # supplies (point == 1), so the worst-case accumulated cost of points
    # meaningfully above 1 exceeds the 2^62 ceiling and solve() must refuse
    # rather than risk a silent int64 wrap -- even for points (like [5, 10]
    # here) whose actual cost would have been fine.  Callers that need such
    # points declare them via set_flow_budget().
    W = make_network()
    with pytest.raises(OverflowError, match="set_flow_budget"):
        W.solve([196.0, 0.44])
    with pytest.raises(OverflowError, match="set_flow_budget"):
        W.solve([5.0, 10.0])


def test_budget_widens_sizing_and_point_computes_correctly():
    W = make_network(budget=6600.0)
    unbudgeted_scale = make_network().scale_factor()
    assert W.scale_factor() < unbudgeted_scale
    W.solve([196.0, 0.44])
    # t1 surplus 10*196-50=1910, emp surplus 150-15*0.44=143.4, all at cost 100.
    assert W.total_cost() == pytest.approx(205340.0, rel=1e-6)
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
