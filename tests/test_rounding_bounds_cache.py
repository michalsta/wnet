"""The cached cost-rounding bounds must follow every change of costs or scale.

cost_round_up_ratio / cost_round_down_ratio / cost_rounding_slack_scaled are
computed once per quantisation and reused by every dual_cut(); after a
position update on a chain, or an intensity rescale, they must equal those of
a network built fresh at the new positions / scale.

Dense networks keep reporting the bounds of their build-time matching costs
after update_positions_and_solve() (the subgraph has no current real cost per
matching arc); that predates the cache and is pinned here as an expected
failure so a fix shows up.
"""

import numpy as np
import pytest

from wnet import Distribution, WassersteinNetwork
from wnet.distances import DistanceMetric
from wnet.wnet_cpp import NetworkSimplex, SlopeDP


def build(positions_base, chain, solver, scale=1000.0):
    rng = np.random.default_rng(1)
    dim = 1 if chain else 2
    base = Distribution(positions_base, rng.uniform(0.5, 1.5, size=positions_base.shape[1]))
    tpos = np.sort(rng.uniform(0, 10, size=(dim, 9)), axis=1)
    target = Distribution(tpos, rng.uniform(0.5, 1.5, size=9))
    kwargs = {"split_distance": 4.0} if chain else {"max_distance": 4.0}
    g = WassersteinNetwork(base, [target], DistanceMetric.L1, solver=solver,
                           intensity_scale=scale, round_max_distance=False, **kwargs)
    g.set_cost_scaling(0)
    g.add_independent_asymmetric_trash(0.7, 0.9)
    g.build()
    return g, target


KEYS = ("cost_rounding_error", "cost_round_up_ratio", "transfer_round_up_ratio",
        "transfer_round_down_ratio", "transfer_absolute_error")


def bounds(g):
    cut = g.dual_cut()
    return tuple(float(cut[k]) for k in KEYS)


def base_positions(chain):
    rng = np.random.default_rng(2)
    return np.sort(rng.uniform(0, 10, size=(1 if chain else 2, 12)), axis=1)


@pytest.mark.parametrize("chain,solver", [(False, NetworkSimplex()), (True, SlopeDP())])
def test_bounds_follow_rescales(chain, solver):
    pos = base_positions(chain)
    g, _ = build(pos, chain, solver)
    g.solve([0.8])
    first = bounds(g)
    assert bounds(g) == first  # repeated queries hit the cache
    g.refine_intensity_precision(10)
    g.solve([0.8])
    fresh, _ = build(pos, chain, solver, scale=10_000.0)
    fresh.solve([0.8])
    assert g.scale_factor() == fresh.scale_factor()
    assert bounds(g) == bounds(fresh)


def test_chain_bounds_follow_position_updates():
    pos = base_positions(True)
    g, target = build(pos, True, SlopeDP())
    g.solve([0.8])
    moved = pos + 0.0007  # keeps the merged chain order
    g.update_positions_and_solve(Distribution(moved, g._base_distribution.intensities), [target])
    fresh, _ = build(moved, True, SlopeDP())
    fresh.solve([0.8])
    assert bounds(g) == bounds(fresh)


@pytest.mark.xfail(strict=True, reason="dense rounding bounds use build-time matching costs after a position update")
def test_dense_bounds_follow_position_updates():
    pos = base_positions(False)
    g, target = build(pos, False, NetworkSimplex())
    g.solve([0.8])
    moved = pos + 0.3
    g.update_positions_and_solve(Distribution(moved, g._base_distribution.intensities), [target])
    fresh, _ = build(moved, False, NetworkSimplex())
    fresh.solve([0.8])
    assert bounds(g) == pytest.approx(bounds(fresh), rel=1e-9, abs=1e-12)
