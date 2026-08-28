"""
Tests for the explicit chain-vs-dense distance-cap semantics.

``max_distance`` is a per-pair matching threshold (dense semantics,
guaranteed); the 1D chain factory may implement it only when provably
equivalent (no cap, or both-side trash rendering the cap inactive).
``split_distance`` requests chain semantics by name: a component-splitting
radius on the merged 1D sequence, within which mass may legally ride the
chain arbitrarily far (multi-hop).
"""

import numpy as np
import pytest

from wnet import WassersteinNetwork
from wnet.distribution import Distribution_1D
from wnet.distances import DistanceMetric
from wnet.wnet_cpp import CostScaling, CapacityScaling, SlopeDP
from wnet.distribution import Distribution


def d1(pos, intens):
    return Distribution_1D(np.array(pos, dtype=float), np.array(intens, dtype=float))


# The canonical divergence example: emp at {0, 5}, theo at {10} wanting both.
# With cap 6, dense forbids the (0 -> 10) pair (distance 10 > 6); the chain
# lets that mass ride through the peak at 5.  Simple trash 100.
EMP_POS, EMP_INT = [0.0, 5.0], [10.0, 10.0]
THEO_POS, THEO_INT = [10.0], [20.0]
TRASH = 100.0
# Dense, cap 6: emp@5 matches theo (10 units at distance 5 = 50); emp@0 has no
# pair within the cap, becomes an isolated dead-end and is trashed (1000); the
# unfilled half of theo's supply is trashed inside the subgraph (1000).
DENSE_COST = 2050.0
# Chain: 10 units at distance 5 + 10 units riding to distance 10 = 150.
CHAIN_COST = 150.0


def _build(max_distance=None, split_distance=None, force_dense_1d=False, solver=None,
           trash=TRASH):
    kw = {}
    if solver is not None:
        kw["solver"] = solver
    net = WassersteinNetwork(
        d1(EMP_POS, EMP_INT),
        [d1(THEO_POS, THEO_INT)],
        DistanceMetric.L1,
        max_distance=max_distance,
        split_distance=split_distance,
        force_dense_1d=force_dense_1d,
        **kw,
    )
    if trash is not None:
        net.add_simple_trash(trash)
    net.build()
    net.solve([1.0])
    return net


def test_tight_cap_max_distance_matches_dense():
    # A tight cap with trash beyond the gate threshold must yield the DENSE
    # per-pair semantics regardless of factory: emp@0 cannot match theo@10.
    chain_free = _build(max_distance=6)
    forced_dense = _build(max_distance=6, force_dense_1d=True)
    assert forced_dense.total_cost() == DENSE_COST
    assert chain_free.total_cost() == DENSE_COST
    # The gate must have chosen the dense factory (2 * trash = 200 > 6).
    assert chain_free.count_chain_edges() == 0


def test_split_distance_reproduces_chain_semantics():
    # split_distance names chain semantics: mass may ride through the
    # intermediate peak beyond the split radius.
    net = _build(split_distance=6)
    assert net.count_chain_edges() > 0
    assert net.total_cost() == CHAIN_COST


def test_generous_cap_uses_chain_and_matches_dense():
    # Cap 250 >= 2 * trash = 200: transport beyond the cap is dominated by
    # double-trashing, so the chain factory is provably equivalent and may be
    # used invisibly.  Cost must equal the forced-dense build bit for bit.
    gated = _build(max_distance=250)
    forced_dense = _build(max_distance=250, force_dense_1d=True)
    assert gated.count_chain_edges() > 0  # gate chose the chain factory
    assert forced_dense.count_chain_edges() == 0
    assert gated.total_cost() == forced_dense.total_cost()


def test_no_cap_uses_chain_and_matches_dense():
    gated = _build(max_distance=None)
    forced_dense = _build(max_distance=None, force_dense_1d=True)
    assert gated.count_chain_edges() > 0
    assert gated.total_cost() == forced_dense.total_cost()


def test_asymmetric_trash_gate():
    # Asymmetric gate threshold is t_exp + t_theo.
    def build_asym(md, te, tt):
        net = WassersteinNetwork(
            d1(EMP_POS, EMP_INT),
            [d1(THEO_POS, THEO_INT)],
            DistanceMetric.L1,
            max_distance=md,
        )
        net.add_experimental_trash(te)
        net.add_theoretical_trash(tt)
        net.build()
        return net

    assert build_asym(100, 30.0, 40.0).count_chain_edges() > 0  # 100 >= 70
    assert build_asym(50, 30.0, 40.0).count_chain_edges() == 0  # 50 < 70


def test_one_sided_trash_never_gates_chain():
    # Only both-side trash makes the cap provably inactive.
    net = WassersteinNetwork(
        d1(EMP_POS, EMP_INT),
        [d1(THEO_POS, THEO_INT)],
        DistanceMetric.L1,
        max_distance=10_000,
    )
    net.add_experimental_trash(1.0)
    net.build()
    assert net.count_chain_edges() == 0


def test_both_caps_rejected():
    with pytest.raises(ValueError, match="not both"):
        WassersteinNetwork(
            d1(EMP_POS, EMP_INT),
            [d1(THEO_POS, THEO_INT)],
            DistanceMetric.L1,
            max_distance=6,
            split_distance=6,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"force_dense_1d": True},
        {"p": 2.0},
        {"solver": CostScaling()},
        {"solver": CapacityScaling()},
    ],
)
def test_split_distance_incompatible_configs_raise(kwargs):
    # split_distance asks for chain semantics by name: impossible requests
    # must raise, never silently fall back to dense.
    p = kwargs.pop("p", 1.0)
    net = WassersteinNetwork(
        d1(EMP_POS, EMP_INT),
        [d1(THEO_POS, THEO_INT)],
        DistanceMetric.L1,
        split_distance=6,
        p=p,
        **kwargs,
    )
    net.add_simple_trash(TRASH)
    with pytest.raises(ValueError, match="split_distance"):
        net.build()


def test_split_distance_rejects_2d():
    base = Distribution(np.array([[0.0, 5.0], [0.0, 5.0]]), np.array([1.0, 1.0]))
    target = Distribution(np.array([[1.0], [1.0]]), np.array([2.0]))
    net = WassersteinNetwork(
        base, [target], DistanceMetric.L2, split_distance=6, round_max_distance=False
    )
    net.add_simple_trash(10)
    with pytest.raises(ValueError, match="1D"):
        net.build()


def test_slopedp_requires_provable_chain():
    # Tight cap, gate fails -> SlopeDP cannot run; the error must point at
    # split_distance.
    net = WassersteinNetwork(
        d1(EMP_POS, EMP_INT),
        [d1(THEO_POS, THEO_INT)],
        DistanceMetric.L1,
        max_distance=6,
        solver=SlopeDP(),
    )
    net.add_simple_trash(TRASH)
    with pytest.raises(ValueError, match="split_distance"):
        net.build()


def test_slopedp_with_split_distance_matches_chain():
    net = _build(split_distance=6, solver=SlopeDP())
    assert net.total_cost() == CHAIN_COST


def test_slopedp_with_generous_cap_gate():
    # Gate passes (cap >= 2 * trash) -> SlopeDP runs on the invisible chain
    # and matches NetworkSimplex.
    ns = _build(max_distance=250)
    dp = _build(max_distance=250, solver=SlopeDP())
    assert dp.total_cost() == ns.total_cost()


def test_trash_added_after_init_feeds_the_gate():
    # The gate decision runs at build() so it sees trash declared after
    # __init__ — the same network flips factory purely on the trash cost.
    def build_with_trash(t):
        net = WassersteinNetwork(
            d1(EMP_POS, EMP_INT),
            [d1(THEO_POS, THEO_INT)],
            DistanceMetric.L1,
            max_distance=100,
        )
        net.add_simple_trash(t)
        net.build()
        return net

    assert build_with_trash(50.0).count_chain_edges() > 0  # 100 >= 2*50
    assert build_with_trash(60.0).count_chain_edges() == 0  # 100 < 2*60
