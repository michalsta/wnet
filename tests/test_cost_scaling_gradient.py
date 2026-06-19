"""Regression: spectrum_proportion_derivatives() must equal d(total_cost)/dw
in COST-SCALING mode (set_cost_scaling), not only in the legacy truncation mode.

The existing derivative suite (test_signal_part_derivatives.py,
test_gradient_cache_fast_approx.py) only builds truncated, integer-intensity
networks with simple trash, so it never exercised the float-backend cost-scaling
path that wnetdeconv uses (real fractional distances via set_cost_scaling).  A
regression there (wnet 3eddf47, "More scaling cleanup work") made the analytic
gradient disagree with finite differences whenever cost scaling is engaged,
while the truncation path stayed correct -- which is why nothing caught it.

These tests assert the gradient matches a central finite difference of
total_cost() with cost scaling ON.
"""
import numpy as np
import pytest

from wnet import WassersteinNetwork
from wnet.distribution import Distribution
from wnet.distances import DistanceMetric


def _build(theo_pos, max_distance, exp_trash, theo_trash,
           intensity_scale=100.0, cost_scaling=True, method=None):
    """One empirical peak at 0 (mass 1); one theoretical spectrum whose peaks are
    at `theo_pos` (uniform mass).  Asymmetric trash.  Returns the network."""
    emp = Distribution(np.array([[0.0]]), np.array([1.0]))
    # Normalised theoretical spectrum (mass 1), matching how wnetdeconv feeds it.
    theo = Distribution(np.array([theo_pos], dtype=float),
                        np.full(len(theo_pos), 1.0 / len(theo_pos)))
    net = WassersteinNetwork(
        emp, [theo], DistanceMetric.LINF, max_distance=max_distance,
        intensity_scale=intensity_scale, round_max_distance=False, method=method,
    )
    if cost_scaling:
        net.set_cost_scaling(0)  # auto cost scale (real fractional distances)
    net.add_experimental_trash(exp_trash)
    net.add_theoretical_trash(theo_trash)
    net.build()
    return net


def _grad_and_fd(net, w=0.5, h=0.2):
    net.solve([w])
    grad = float(net.spectrum_proportion_derivatives()[0])
    net.solve([w + h])
    c_plus = net.total_cost()
    net.solve([w - h])
    c_minus = net.total_cost()
    fd = (c_plus - c_minus) / (2 * h)
    return grad, fd


@pytest.mark.parametrize("d,exp_trash,theo_trash", [
    (1.0, 3.0, 3.0),
    (2.0, 3.0, 3.0),
    (1.0, 5.0, 3.0),
    (0.1, 0.3, 0.3),
])
def test_proportion_gradient_matches_fd_single_peak(d, exp_trash, theo_trash):
    net = _build([-d], max_distance=10.0, exp_trash=exp_trash, theo_trash=theo_trash)
    grad, fd = _grad_and_fd(net)
    assert grad == pytest.approx(fd, abs=1e-6), (
        f"cost-scaling gradient {grad} != finite diff {fd} (d={d})"
    )


def test_proportion_gradient_matches_fd_many_peaks():
    # Several peaks clustered against the single empirical peak: the gradient
    # must stay ~constant (true d(cost)/dw), not grow with the peak count.
    # intensity_scale must resolve the per-peak supply (mass 1/N) -- 100 would
    # round each peak's supply to zero and make the problem degenerate.
    net = _build(list(np.linspace(-0.01, 0.01, 200)),
                 max_distance=0.5, exp_trash=0.3, theo_trash=0.3,
                 intensity_scale=1e6)
    grad, fd = _grad_and_fd(net)
    # Looser tolerance than the single-peak cases: with many clustered peaks the
    # cost is mildly curved, so the central difference (h=0.2) carries ~1e-4
    # discretisation error.  The point is the gradient is ~ -0.29 (true value),
    # not the pre-fix -2..-1820 that grew with the peak count.
    assert grad == pytest.approx(fd, rel=1e-2), (
        f"cost-scaling gradient {grad} != finite diff {fd} (200 peaks)"
    )


def test_truncation_path_unaffected():
    # Sanity: the legacy truncation path (integer distances, no cost scaling)
    # has always been correct and must stay correct.
    net = _build([-1.0], max_distance=10.0, exp_trash=3.0, theo_trash=3.0,
                 cost_scaling=False)
    grad, fd = _grad_and_fd(net)
    assert grad == pytest.approx(fd, abs=1e-6)
