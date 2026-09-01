"""
Tests for asymmetric (experimental and theoretical) trash edges.

Topology reminder:
  Source → EmpiricalNode  → TheoreticalNode → Sink   (normal path)
  EmpiricalNode → Sink                               (experimental trash)
  Source → TheoreticalNode                           (theoretical trash)

The two asymmetric types are mutually exclusive with SimpleTrashEdge.
"""

import pytest
import numpy as np

from wnet import Distribution, WassersteinNetwork
from wnet.distances import DistanceMetric


def _dist1d(positions, intensities):
    return Distribution(
        np.array([positions], dtype=np.float64),
        np.array(intensities, dtype=np.int64),
    )


def _solve(wnet, point=None):
    wnet.build()
    wnet.solve() if point is None else wnet.solve(point)
    return wnet.total_cost()


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------


def test_experimental_trash_excess_empirical():
    # emp=7, theo=4, dist=1 → 4 match + 3 exp-trash
    # cost = 4*1 + 3*10 = 34
    emp = _dist1d([0], [7])
    theo = _dist1d([1], [4])
    W = WassersteinNetwork(emp, [theo], DistanceMetric.L1, 100)
    W.add_experimental_trash(10)
    assert _solve(W) == 34


def test_theoretical_trash_excess_theoretical():
    # emp=3, theo=7, dist=1 → 3 match + 4 theo-trash
    # cost = 3*1 + 4*10 = 43
    emp = _dist1d([0], [3])
    theo = _dist1d([1], [7])
    W = WassersteinNetwork(emp, [theo], DistanceMetric.L1, 100)
    W.add_theoretical_trash(10)
    assert _solve(W) == 43


def test_both_trash_types_excess_empirical():
    # E > T → only experimental trash should fire
    # cost = 4*1 + 3*5 = 19
    emp = _dist1d([0], [7])
    theo = _dist1d([1], [4])
    W = WassersteinNetwork(emp, [theo], DistanceMetric.L1, 100)
    W.add_experimental_trash(5)
    W.add_theoretical_trash(5)
    assert _solve(W) == 19


def test_both_trash_types_excess_theoretical():
    # T > E → only theoretical trash should fire
    # cost = 3*1 + 4*5 = 23
    emp = _dist1d([0], [3])
    theo = _dist1d([1], [7])
    W = WassersteinNetwork(emp, [theo], DistanceMetric.L1, 100)
    W.add_experimental_trash(5)
    W.add_theoretical_trash(5)
    assert _solve(W) == 23


def test_both_trash_types_balanced():
    # E = T = 4, dist=1 → all 4 units match, no trash used
    # cost = 4*1 = 4
    emp = _dist1d([0], [4])
    theo = _dist1d([1], [4])
    W = WassersteinNetwork(emp, [theo], DistanceMetric.L1, 100)
    W.add_experimental_trash(10)
    W.add_theoretical_trash(10)
    assert _solve(W) == 4


def test_all_trash_matching_too_expensive():
    # dist=1000, trash=10; optimizer trashes all empirical signal
    # emp=5, theo=5, exp_cost=10 → 5 * 10 = 50 (exp trash beats matching)
    emp = _dist1d([0], [5])
    theo = _dist1d([1000], [5])
    W = WassersteinNetwork(emp, [theo], DistanceMetric.L1, 10_000)
    W.add_experimental_trash(10)
    W.add_theoretical_trash(20)  # more expensive, should not be used
    assert _solve(W) == 50


def test_theoretical_trash_preferred_when_cheaper():
    # emp=3, theo=7, dist=1000, exp_trash=10, theo_trash=8
    # supply = 7; SrcToEmpiricalEdge cap=3 but flow through it is not mandatory.
    # Cheapest path: all 7 units via Source→Theo→Sink (theo-trash=8) = 56.
    # Empirical nodes carry 0 flow — cheaper than mixing exp-trash (10) in.
    emp = _dist1d([0], [3])
    theo = _dist1d([1000], [7])
    W = WassersteinNetwork(emp, [theo], DistanceMetric.L1, 10_000)
    W.add_experimental_trash(10)
    W.add_theoretical_trash(8)
    assert _solve(W) == 56


# ---------------------------------------------------------------------------
# Cross-validation against simple trash
# ---------------------------------------------------------------------------


def test_exp_trash_equals_simple_trash_excess_empirical():
    # When E > T, experimental trash and simple trash with equal cost should agree.
    # emp=7, theo=4, dist=1, cost=10 → 34 either way
    emp = _dist1d([0], [7])
    theo = _dist1d([1], [4])

    W_asym = WassersteinNetwork(emp, [theo], DistanceMetric.L1, 100)
    W_asym.add_experimental_trash(10)
    cost_asym = _solve(W_asym)

    W_sym = WassersteinNetwork(emp, [theo], DistanceMetric.L1, 100)
    W_sym.add_simple_trash(10)
    cost_sym = _solve(W_sym)

    assert cost_asym == cost_sym


def test_theo_trash_equals_simple_trash_excess_theoretical():
    # When T > E, theoretical trash and simple trash with equal cost should agree.
    # emp=3, theo=7, dist=1, cost=10 → 43 either way
    emp = _dist1d([0], [3])
    theo = _dist1d([1], [7])

    W_asym = WassersteinNetwork(emp, [theo], DistanceMetric.L1, 100)
    W_asym.add_theoretical_trash(10)
    cost_asym = _solve(W_asym)

    W_sym = WassersteinNetwork(emp, [theo], DistanceMetric.L1, 100)
    W_sym.add_simple_trash(10)
    cost_sym = _solve(W_sym)

    assert cost_asym == cost_sym


def test_both_trash_equals_simple_trash_two_components():
    # Multi-component graph: component 1 has E > T, component 2 has T > E.
    # Points separated by 1000 → two independent subgraphs.
    # Component 1: emp=7 @ 0, theo=4 @ 1 → 4 matched at distance 1
    # Component 2: emp=3 @ 1000, theo=7 @ 1001 → 3 matched at distance 1
    #
    # Both models annihilate, and the excesses sit on opposite sides in the
    # two components: 3 unmatched empirical units against 4 unmatched
    # theoretical ones.  The bill is priced over the network as a whole —
    # max(10, 11) - 7 = 4 escaping units at 10 — so the split must not charge
    # each component's excess separately (which would give 77).
    emp = _dist1d([0, 1000], [7, 3])
    theo = _dist1d([1, 1001], [4, 7])

    W_asym = WassersteinNetwork(emp, [theo], DistanceMetric.L1, 10)
    W_asym.add_experimental_trash(10)
    W_asym.add_theoretical_trash(10)
    cost_asym = _solve(W_asym)

    W_sym = WassersteinNetwork(emp, [theo], DistanceMetric.L1, 10)
    W_sym.add_simple_trash(10)
    cost_sym = _solve(W_sym)

    assert cost_asym == cost_sym == 4 + 3 + 4 * 10


# ---------------------------------------------------------------------------
# Multiple theoretical spectra
# ---------------------------------------------------------------------------


def test_two_theoretical_spectra_experimental_trash():
    # emp=5 @ 0, theo_A=4 @ 1, theo_B=2 @ 1
    # point=[0.5, 0.5] → integer-scaled intensities: A=2, B=1, total=3
    # 3 match + 2 emp-trash → cost = 3*1 + 2*10 = 23
    emp = _dist1d([0], [5])
    theoA = _dist1d([1], [4])
    theoB = _dist1d([1], [2])
    W = WassersteinNetwork(emp, [theoA, theoB], DistanceMetric.L1, 100)
    W.add_experimental_trash(10)
    W.build()
    W.solve([0.5, 0.5])
    assert W.total_cost() == 23


# ---------------------------------------------------------------------------
# Guard / error handling
# ---------------------------------------------------------------------------


def test_experimental_trash_after_build_raises():
    emp = _dist1d([0], [5])
    theo = _dist1d([1], [5])
    W = WassersteinNetwork(emp, [theo], DistanceMetric.L1, 100)
    W.build()
    with pytest.raises(RuntimeError):
        W.add_experimental_trash(10)


def test_theoretical_trash_after_build_raises():
    emp = _dist1d([0], [5])
    theo = _dist1d([1], [5])
    W = WassersteinNetwork(emp, [theo], DistanceMetric.L1, 100)
    W.build()
    with pytest.raises(RuntimeError):
        W.add_theoretical_trash(10)


def test_experimental_trash_exclusive_with_simple_trash():
    emp = _dist1d([0], [5])
    theo = _dist1d([1], [5])
    W = WassersteinNetwork(emp, [theo], DistanceMetric.L1, 100)
    W.add_simple_trash(10)
    with pytest.raises(RuntimeError):
        W.add_experimental_trash(10)


def test_theoretical_trash_exclusive_with_simple_trash():
    emp = _dist1d([0], [5])
    theo = _dist1d([1], [5])
    W = WassersteinNetwork(emp, [theo], DistanceMetric.L1, 100)
    W.add_simple_trash(10)
    with pytest.raises(RuntimeError):
        W.add_theoretical_trash(10)


def test_simple_trash_exclusive_with_experimental_trash():
    emp = _dist1d([0], [5])
    theo = _dist1d([1], [5])
    W = WassersteinNetwork(emp, [theo], DistanceMetric.L1, 100)
    W.add_experimental_trash(10)
    with pytest.raises(RuntimeError):
        W.add_simple_trash(10)


def test_simple_trash_exclusive_with_theoretical_trash():
    emp = _dist1d([0], [5])
    theo = _dist1d([1], [5])
    W = WassersteinNetwork(emp, [theo], DistanceMetric.L1, 100)
    W.add_theoretical_trash(10)
    with pytest.raises(RuntimeError):
        W.add_simple_trash(10)


def test_experimental_trash_double_add_raises():
    emp = _dist1d([0], [5])
    theo = _dist1d([1], [5])
    W = WassersteinNetwork(emp, [theo], DistanceMetric.L1, 100)
    W.add_experimental_trash(10)
    with pytest.raises(RuntimeError):
        W.add_experimental_trash(10)


def test_theoretical_trash_double_add_raises():
    emp = _dist1d([0], [5])
    theo = _dist1d([1], [5])
    W = WassersteinNetwork(emp, [theo], DistanceMetric.L1, 100)
    W.add_theoretical_trash(10)
    with pytest.raises(RuntimeError):
        W.add_theoretical_trash(10)


def test_signal_part_derivatives_exp_trash_excess_empirical():
    # emp=7 @ 0, theo=4 @ 1, dist=1, C_exp=10.
    # Optimal: 4 match + 3 exp-trash = 34. Theo node is at capacity.
    # Increasing theo intensity by 1 saves one exp-trash unit (10) but pays
    # matching cost (1): net = 1 - 10 = -9. Derivative = -9.
    emp = _dist1d([0], [7])
    theo = _dist1d([1], [4])
    W = WassersteinNetwork(emp, [theo], DistanceMetric.L1, 100, force_dense_1d=True)
    W.add_experimental_trash(10)
    W.build()
    W.solve()
    derivs = W.signal_part_derivatives()
    assert derivs == {0: {0: -9}}


def test_signal_part_derivatives_theo_trash_excess_theoretical():
    # emp=3 @ 0, theo=7 @ 1, dist=1, C_theo=10.
    # Optimal: 3 match + 4 theo-trash = 43. Theo node has 4 units of slack.
    # Adding 1 to theo intensity: supply grows by 1 → one more unit via theo-trash.
    # Derivative = C_theo = 10.
    emp = _dist1d([0], [3])
    theo = _dist1d([1], [7])
    W = WassersteinNetwork(emp, [theo], DistanceMetric.L1, 100, force_dense_1d=True)
    W.add_theoretical_trash(10)
    W.build()
    W.solve()
    derivs = W.signal_part_derivatives()
    assert derivs == {0: {0: 10}}


# ---------------------------------------------------------------------------
# Derivatives — chain factory (1D without force_dense_1d)
# ---------------------------------------------------------------------------


def test_signal_part_derivatives_exp_trash_chain_single_pair():
    # Chain factory. Same setup as the dense test: emp=7@0, theo=4@1, C_exp=10.
    # Derivative = match cost (1) - trash cost (10) = -9.
    emp = _dist1d([0], [7])
    theo = _dist1d([1], [4])
    W = WassersteinNetwork(emp, [theo], DistanceMetric.L1, 100)
    W.add_experimental_trash(10)
    W.build()
    W.solve()
    assert W.signal_part_derivatives() == {0: {0: -9}}


def test_signal_part_derivatives_theo_trash_chain_single_pair():
    # Chain factory. emp=3@0, theo=7@1, C_theo=10.
    # Extra unit must go via theo-trash → derivative = 10.
    emp = _dist1d([0], [3])
    theo = _dist1d([1], [7])
    W = WassersteinNetwork(emp, [theo], DistanceMetric.L1, 100)
    W.add_theoretical_trash(10)
    W.build()
    W.solve()
    assert W.signal_part_derivatives() == {0: {0: 10}}


def test_signal_part_derivatives_chain_matches_dense_exp_trash():
    emp = _dist1d([0], [7])
    theo = _dist1d([1], [4])
    W_chain = WassersteinNetwork(emp, [theo], DistanceMetric.L1, 100)
    W_chain.add_experimental_trash(10)
    W_chain.build()
    W_chain.solve()
    W_dense = WassersteinNetwork(
        emp, [theo], DistanceMetric.L1, 100, force_dense_1d=True
    )
    W_dense.add_experimental_trash(10)
    W_dense.build()
    W_dense.solve()
    assert W_chain.signal_part_derivatives() == W_dense.signal_part_derivatives()


def test_signal_part_derivatives_chain_matches_dense_theo_trash():
    emp = _dist1d([0], [3])
    theo = _dist1d([1], [7])
    W_chain = WassersteinNetwork(emp, [theo], DistanceMetric.L1, 100)
    W_chain.add_theoretical_trash(10)
    W_chain.build()
    W_chain.solve()
    W_dense = WassersteinNetwork(
        emp, [theo], DistanceMetric.L1, 100, force_dense_1d=True
    )
    W_dense.add_theoretical_trash(10)
    W_dense.build()
    W_dense.solve()
    assert W_chain.signal_part_derivatives() == W_dense.signal_part_derivatives()


def test_signal_part_derivatives_exp_trash_three_node_chain():
    # Chain order: emp@0 — theo@5 — emp@10.  C_exp = 20.
    # E=7 > T=2. Optimal: 2 match (dist=5) + 5 exp-trash = 110.
    # Cheapest Sink→theo cycle: Sink → emp@? (cost -20) → theo@5 (chain, cost +5) = -15.
    emp = Distribution(
        np.array([[0, 10]], dtype=np.float64), np.array([3, 4], dtype=np.int64)
    )
    theo = Distribution(
        np.array([[5]], dtype=np.float64), np.array([2], dtype=np.int64)
    )
    W = WassersteinNetwork(emp, [theo], DistanceMetric.L1, 100)
    W.add_experimental_trash(20)
    W.build()
    W.solve()
    assert W.total_cost() == 110
    assert W.signal_part_derivatives() == {0: {0: -15}}


def test_signal_part_derivatives_theo_trash_three_node_chain():
    # Chain order: theo@0 — emp@5 — theo@10.  C_theo = 20.
    # T=7 > E=2. Optimal: 2 match (dist=5) + 5 theo-trash = 110.
    # Extra Source unit goes via theo-trash for either node → derivative = 20.
    emp = Distribution(np.array([[5]], dtype=np.float64), np.array([2], dtype=np.int64))
    theo = Distribution(
        np.array([[0, 10]], dtype=np.float64), np.array([3, 4], dtype=np.int64)
    )
    W = WassersteinNetwork(emp, [theo], DistanceMetric.L1, 100)
    W.add_theoretical_trash(20)
    W.build()
    W.solve()
    assert W.total_cost() == 110
    assert W.signal_part_derivatives() == {0: {0: 20, 1: 20}}


def test_signal_part_derivatives_matches_simple_trash_excess_empirical():
    # When E > T and C_exp = C_simple, derivatives should be identical.
    emp = _dist1d([0], [7])
    theo = _dist1d([1], [4])

    W_asym = WassersteinNetwork(
        emp, [theo], DistanceMetric.L1, 100, force_dense_1d=True
    )
    W_asym.add_experimental_trash(10)
    W_asym.build()
    W_asym.solve()

    W_sym = WassersteinNetwork(emp, [theo], DistanceMetric.L1, 100, force_dense_1d=True)
    W_sym.add_simple_trash(10)
    W_sym.build()
    W_sym.solve()

    assert W_asym.signal_part_derivatives() == W_sym.signal_part_derivatives()


def test_signal_part_derivatives_matches_simple_trash_excess_theoretical():
    # When T > E and C_theo = C_simple, derivatives should be identical.
    emp = _dist1d([0], [3])
    theo = _dist1d([1], [7])

    W_asym = WassersteinNetwork(
        emp, [theo], DistanceMetric.L1, 100, force_dense_1d=True
    )
    W_asym.add_theoretical_trash(10)
    W_asym.build()
    W_asym.solve()

    W_sym = WassersteinNetwork(emp, [theo], DistanceMetric.L1, 100, force_dense_1d=True)
    W_sym.add_simple_trash(10)
    W_sym.build()
    W_sym.solve()

    assert W_asym.signal_part_derivatives() == W_sym.signal_part_derivatives()


# ---------------------------------------------------------------------------
# Marginals under a single asymmetric trash kind
#
# set_point() sizes the flow budget from the escape routes that exist:
# experimental trash alone pins it to the empirical total, theoretical trash
# alone to the theoretical total, and the annihilating models to max(E, T).
# Whether one more theoretical unit costs anything depends on that rule, so the
# marginal must be checked on both sides of E vs T -- reading the annihilating
# comparison unconditionally used to take the wrong branch for the one-sided
# models, silently reporting a marginal that a re-solve contradicts.
# ---------------------------------------------------------------------------


def _one_sided(emp_int, theo_int, mode, cost, dist=1, force_dense=True):
    emp = _dist1d([0], [emp_int])
    theo = _dist1d([dist], [theo_int])
    W = WassersteinNetwork(
        emp, [theo], DistanceMetric.L1, 100, force_dense_1d=force_dense
    )
    if mode == "experimental":
        W.add_experimental_trash(cost)
    else:
        W.add_theoretical_trash(cost)
    W.build()
    W.solve()
    return W


def _finite_difference(emp_int, theo_int, mode, cost, dist=1):
    """True marginal of one extra theoretical unit: re-solve and subtract."""
    base = _one_sided(emp_int, theo_int, mode, cost, dist).total_cost()
    bumped = _one_sided(emp_int, theo_int + 1, mode, cost, dist).total_cost()
    return bumped - base


def test_theoretical_trash_marginal_with_empirical_surplus():
    # E = 3 > T = 2, theoretical trash only, so the budget is T and grows with
    # every added unit: the new unit is matched from the empirical surplus at
    # distance 1.  Reporting 0 here (the fixed-budget branch) contradicts a
    # re-solve, which costs exactly 1 more.
    W = _one_sided(3, 2, "theoretical", 5)
    assert W.total_cost() == 2
    assert W.signal_part_derivatives()[0][0] == 1
    assert _finite_difference(3, 2, "theoretical", 5) == 1


def test_experimental_trash_marginal_with_theoretical_surplus():
    # T = 5 >= E = 3, experimental trash only, so the budget is E and does not
    # move: an extra theoretical unit only widens already-slack capacity.
    W = _one_sided(3, 5, "experimental", 5)
    assert W.signal_part_derivatives()[0][0] == 0
    assert _finite_difference(3, 5, "experimental", 5) == 0


@pytest.mark.parametrize("mode", ["experimental", "theoretical"])
@pytest.mark.parametrize("cost", [2, 10])
@pytest.mark.parametrize(
    "emp_int, theo_int",
    [(1, 5), (2, 4), (3, 3), (4, 2), (5, 1), (7, 4), (4, 7), (6, 6)],
)
def test_one_sided_marginal_matches_resolve(mode, cost, emp_int, theo_int):
    # Sweeps both sides of the E vs T boundary, and both trash costs above and
    # below the matching distance, so neither branch can pass by coincidence.
    W = _one_sided(emp_int, theo_int, mode, cost)
    reported = W.signal_part_derivatives()[0][0]
    assert reported == _finite_difference(emp_int, theo_int, mode, cost)


@pytest.mark.parametrize("mode", ["experimental", "theoretical"])
@pytest.mark.parametrize("emp_int, theo_int", [(3, 2), (2, 3), (6, 6)])
def test_one_sided_proportion_derivative_matches_marginals(mode, emp_int, theo_int):
    # The proportion derivative weights the same per-unit marginals by the real
    # theoretical intensities, so the two must stay consistent.
    W = _one_sided(emp_int, theo_int, mode, 5)
    per_peak = W.signal_part_derivatives()[0][0]
    assert W.spectrum_proportion_derivatives()[0] == per_peak * theo_int


@pytest.mark.parametrize("mode", ["experimental", "theoretical"])
@pytest.mark.parametrize("emp_int, theo_int", [(3, 2), (2, 3), (5, 5)])
def test_one_sided_marginal_agrees_across_factories(mode, emp_int, theo_int):
    # The chain and dense factories share the marginal dispatch; both must
    # report the marginal a re-solve confirms.
    dense = _one_sided(emp_int, theo_int, mode, 5, force_dense=True)
    chain = _one_sided(emp_int, theo_int, mode, 5, force_dense=False)
    truth = _finite_difference(emp_int, theo_int, mode, 5)
    assert dense.signal_part_derivatives()[0][0] == truth
    assert chain.signal_part_derivatives()[0][0] == truth


def test_experimental_trash_marginal_can_be_negative():
    # E = 2 at 0; theoretical 1 unit at distance 1 and 1 unit at distance 2;
    # experimental trash 3.  The budget is E and does not move, so an extra
    # theoretical unit cannot be routed in from the source.  What it buys is
    # re-routing the empirical unit currently matched at distance 2 onto the
    # nearer peak, which is worth -1.  The growing-budget branch looks for a
    # source augmentation, finds none, and reports 0.
    emp = _dist1d([0], [2])

    def solved(theo_int):
        W = WassersteinNetwork(
            emp,
            [_dist1d([1, 2], theo_int)],
            DistanceMetric.L1,
            100,
            force_dense_1d=True,
        )
        W.add_experimental_trash(3)
        W.build()
        W.solve()
        return W

    W = solved([1, 1])
    base = W.total_cost()
    assert base == 3  # 1 unit at distance 1, 1 unit at distance 2
    derivs = W.signal_part_derivatives()[0]
    assert derivs[0] == solved([2, 1]).total_cost() - base == -1
    assert derivs[1] == solved([1, 2]).total_cost() - base == 0
