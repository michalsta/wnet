"""
Tests for asymmetric (experimental and theoretical) trash edges.

Topology reminder:
  Source → EmpiricalNode  → TheoreticalNode → Sink   (normal path)
  EmpiricalNode → Sink                               (experimental trash)
  Source → TheoreticalNode                           (theoretical trash)
  Source → Sink                                      (simple trash)

All three trash types may be combined freely: simple + experimental,
simple + theoretical, experimental + theoretical, or all three.  The MCF
solver picks the cheapest available escape for each unit.
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
    # Multi-component graph with opposing imbalance per component:
    #   component 1: E > T (emp excess 3)
    #   component 2: T > E (theo excess 4)
    # Under the global cost model the subgraph decomposition is invisible:
    # cost = match + global trash flow * unit cost.
    # E_total=10, T_total=11, match=4+3=7, supply=11, trash=4. All routes
    # at C=10 in both setups, so cost = 7 + 40 = 47 either way.
    emp = _dist1d([0, 1000], [7, 3])
    theo = _dist1d([1, 1001], [4, 7])

    W_asym = WassersteinNetwork(emp, [theo], DistanceMetric.L1, 10)
    W_asym.add_experimental_trash(10)
    W_asym.add_theoretical_trash(10)
    cost_asym = _solve(W_asym)

    W_sym = WassersteinNetwork(emp, [theo], DistanceMetric.L1, 10)
    W_sym.add_simple_trash(10)
    cost_sym = _solve(W_sym)

    assert cost_asym == cost_sym == 47


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


# ---------------------------------------------------------------------------
# Combined simple + asymmetric trash
# ---------------------------------------------------------------------------


def test_simple_plus_exp_uses_cheaper_simple():
    # E > T, exp_trash=100, simple_trash=10.  Simple is cheaper for excess emp.
    # cost = 4*1 + 3*10 = 34.
    emp = _dist1d([0], [7])
    theo = _dist1d([1], [4])
    W = WassersteinNetwork(emp, [theo], DistanceMetric.L1, 100)
    W.add_simple_trash(10)
    W.add_experimental_trash(100)
    assert _solve(W) == 34


def test_simple_plus_exp_uses_cheaper_exp():
    # E > T, simple=100, exp=10.  Exp is cheaper.
    # cost = 4*1 + 3*10 = 34.
    emp = _dist1d([0], [7])
    theo = _dist1d([1], [4])
    W = WassersteinNetwork(emp, [theo], DistanceMetric.L1, 100)
    W.add_simple_trash(100)
    W.add_experimental_trash(10)
    assert _solve(W) == 34


def test_simple_plus_theo_uses_cheaper_theo():
    # T > E, simple=100, theo=10.  Theo is cheaper for excess theo.
    # cost = 3*1 + 4*10 = 43.
    emp = _dist1d([0], [3])
    theo = _dist1d([1], [7])
    W = WassersteinNetwork(emp, [theo], DistanceMetric.L1, 100)
    W.add_simple_trash(100)
    W.add_theoretical_trash(10)
    assert _solve(W) == 43


def test_all_three_trash_types_simple_cheapest():
    # E > T, simple=5 cheapest, exp=10, theo=10.
    # cost = 4*1 + 3*5 = 19.
    emp = _dist1d([0], [7])
    theo = _dist1d([1], [4])
    W = WassersteinNetwork(emp, [theo], DistanceMetric.L1, 100)
    W.add_simple_trash(5)
    W.add_experimental_trash(10)
    W.add_theoretical_trash(10)
    assert _solve(W) == 19


def test_all_three_trash_types_asym_cheapest():
    # E > T, simple=100, exp=5, theo=100.  Exp is cheapest for excess emp.
    # cost = 4*1 + 3*5 = 19.
    emp = _dist1d([0], [7])
    theo = _dist1d([1], [4])
    W = WassersteinNetwork(emp, [theo], DistanceMetric.L1, 100)
    W.add_simple_trash(100)
    W.add_experimental_trash(5)
    W.add_theoretical_trash(100)
    assert _solve(W) == 19


# ---------------------------------------------------------------------------
# Global-model invariance: the subgraph decomposition is purely a
# computational optimization, so results must be invariant to whether
# max_distance forces a split or keeps everything in a single subgraph.
# ---------------------------------------------------------------------------


def _build_and_solve(emp_d, theo_d, max_dist, trash):
    W = WassersteinNetwork(emp_d, [theo_d], DistanceMetric.L1, max_dist, force_dense_1d=True)
    if "simple" in trash:
        W.add_simple_trash(trash["simple"])
    if "exp" in trash:
        W.add_experimental_trash(trash["exp"])
    if "theo" in trash:
        W.add_theoretical_trash(trash["theo"])
    W.build()
    W.solve()
    return W


@pytest.mark.parametrize("trash", [
    {"simple": 10},
    {"exp": 10, "theo": 10},
    {"simple": 10, "exp": 10, "theo": 10},
    {"simple": 100, "exp": 10, "theo": 100},
    {"simple": 5, "exp": 100, "theo": 100},
])
def test_global_cost_invariant_to_decomposition_opposing(trash):
    # Opposing imbalance: sg1 has E>T, sg2 has T>E.  Split decomposition
    # must give the same total_cost as a single-graph solve.
    emp = _dist1d([0, 100], [5, 2])
    theo = _dist1d([1, 101], [2, 6])
    W_split = _build_and_solve(emp, theo, 10, trash)
    W_global = _build_and_solve(emp, theo, 1000, trash)
    assert W_split.total_cost() == W_global.total_cost()


@pytest.mark.parametrize("trash", [
    {"simple": 10},
    {"exp": 10, "theo": 10},
    {"simple": 10, "exp": 10, "theo": 10},
    {"simple": 100, "exp": 10, "theo": 100},
    {"simple": 5, "exp": 100, "theo": 100},
])
def test_global_derivatives_invariant_to_decomposition_opposing(trash):
    # Same setup: derivatives must be invariant to the decomposition.
    emp = _dist1d([0, 100], [5, 2])
    theo = _dist1d([1, 101], [2, 6])
    W_split = _build_and_solve(emp, theo, 10, trash)
    W_global = _build_and_solve(emp, theo, 1000, trash)
    assert W_split.signal_part_derivatives() == W_global.signal_part_derivatives()


def test_global_invariance_dead_end_peaks():
    # E peaks at one location, T peaks far away → all peaks become
    # dead-ends.  Cost = max(E,T) * cheapest trash, not E+T * cheapest.
    emp = _dist1d([0], [6])
    theo = _dist1d([1000], [4])
    W = WassersteinNetwork(emp, [theo], DistanceMetric.L1, 10)
    W.add_simple_trash(10)
    W.build(); W.solve()
    # supply = max(6, 4) = 6, all via simple at 10. match = 0.
    assert W.total_cost() == 6 * 10


def test_simple_plus_asym_decomposed_subgraphs():
    # Two disjoint components with opposing imbalance.  Combined trash is
    # added to each subgraph; the global model pairs the cross-subgraph
    # excess via the simple trash arc:
    #   E_total=10, T_total=11, match=7, supply=11, trash=4.
    #   Simple (5) is cheapest, exp/theo at 100 unused.
    #   cost = 7 (match) + 4*5 (simple) = 27.
    emp = _dist1d([0, 1000], [7, 3])
    theo = _dist1d([1, 1001], [4, 7])
    W = WassersteinNetwork(emp, [theo], DistanceMetric.L1, 10)
    W.add_simple_trash(5)
    W.add_experimental_trash(100)
    W.add_theoretical_trash(100)
    assert _solve(W) == 27


def test_combined_trash_matches_simple_only_when_simple_cheaper():
    # When simple is uniformly cheaper than any asym, results must match
    # simple-only configuration.
    emp = _dist1d([0, 1000], [7, 3])
    theo = _dist1d([1, 1001], [4, 7])

    W_combined = WassersteinNetwork(emp, [theo], DistanceMetric.L1, 10)
    W_combined.add_simple_trash(5)
    W_combined.add_experimental_trash(100)
    W_combined.add_theoretical_trash(100)
    cost_combined = _solve(W_combined)

    W_simple = WassersteinNetwork(emp, [theo], DistanceMetric.L1, 10)
    W_simple.add_simple_trash(5)
    cost_simple = _solve(W_simple)

    assert cost_combined == cost_simple


def test_signal_part_derivatives_simple_plus_asym_excess_empirical():
    # E=7, T=4, simple=100, exp=10.  Exp is cheaper for emp side.
    # Optimal: 4 match + 3 exp-trash = 34.  Add 1 to T: 5 match + 2 exp = 25,
    # delta = -9.  Same as exp-only derivative.
    emp = _dist1d([0], [7])
    theo = _dist1d([1], [4])
    W = WassersteinNetwork(emp, [theo], DistanceMetric.L1, 100, force_dense_1d=True)
    W.add_simple_trash(100)
    W.add_experimental_trash(10)
    W.build()
    W.solve()
    assert W.signal_part_derivatives() == {0: {0: -9}}


def test_signal_part_derivatives_simple_plus_asym_excess_theoretical():
    # E=3, T=7, simple=100, theo=10.  Theo cheaper for theo side.
    # Adding 1 to theo intensity adds 1 to supply → 1 more theo-trash = +10.
    emp = _dist1d([0], [3])
    theo = _dist1d([1], [7])
    W = WassersteinNetwork(emp, [theo], DistanceMetric.L1, 100, force_dense_1d=True)
    W.add_simple_trash(100)
    W.add_theoretical_trash(10)
    W.build()
    W.solve()
    assert W.signal_part_derivatives() == {0: {0: 10}}


def test_signal_part_derivatives_combined_matches_asym_only_when_asym_cheaper():
    # When asym trash is cheaper than simple, the derivative must match the
    # asym-only configuration.
    emp = _dist1d([0], [7])
    theo = _dist1d([1], [4])

    W_combined = WassersteinNetwork(
        emp, [theo], DistanceMetric.L1, 100, force_dense_1d=True
    )
    W_combined.add_simple_trash(1000)
    W_combined.add_experimental_trash(10)
    W_combined.build()
    W_combined.solve()

    W_asym = WassersteinNetwork(
        emp, [theo], DistanceMetric.L1, 100, force_dense_1d=True
    )
    W_asym.add_experimental_trash(10)
    W_asym.build()
    W_asym.solve()

    assert W_combined.signal_part_derivatives() == W_asym.signal_part_derivatives()


def test_signal_part_derivatives_combined_matches_simple_only_when_simple_cheaper():
    # When simple is cheaper than asym, the derivative must match the
    # simple-only configuration.
    emp = _dist1d([0], [7])
    theo = _dist1d([1], [4])

    W_combined = WassersteinNetwork(
        emp, [theo], DistanceMetric.L1, 100, force_dense_1d=True
    )
    W_combined.add_simple_trash(10)
    W_combined.add_experimental_trash(1000)
    W_combined.build()
    W_combined.solve()

    W_simple = WassersteinNetwork(
        emp, [theo], DistanceMetric.L1, 100, force_dense_1d=True
    )
    W_simple.add_simple_trash(10)
    W_simple.build()
    W_simple.solve()

    assert W_combined.signal_part_derivatives() == W_simple.signal_part_derivatives()


def test_combined_trash_chain_factory_matches_dense():
    # Chain (1D) and dense factories should give the same total cost and the
    # same derivatives when both simple + asym are active.
    emp = _dist1d([0], [7])
    theo = _dist1d([1], [4])

    W_chain = WassersteinNetwork(emp, [theo], DistanceMetric.L1, 100)
    W_chain.add_simple_trash(50)
    W_chain.add_experimental_trash(10)
    W_chain.build()
    W_chain.solve()

    W_dense = WassersteinNetwork(
        emp, [theo], DistanceMetric.L1, 100, force_dense_1d=True
    )
    W_dense.add_simple_trash(50)
    W_dense.add_experimental_trash(10)
    W_dense.build()
    W_dense.solve()

    assert W_chain.total_cost() == W_dense.total_cost()
    assert W_chain.signal_part_derivatives() == W_dense.signal_part_derivatives()


def test_update_positions_and_get_gradient_combined_trash():
    # Position-gradient flow path must work with combined trash.  Use a 2D
    # case so we exercise the dense (non-chain) gradient.
    emp = Distribution(
        np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64),
        np.array([5, 5], dtype=np.int64),
    )
    theo = Distribution(
        np.array([[0.5, 0.0], [1.5, 0.0]], dtype=np.float64),
        np.array([5, 5], dtype=np.int64),
    )
    W = WassersteinNetwork(emp, [theo], DistanceMetric.L2, 100)
    W.add_simple_trash(50)
    W.add_experimental_trash(50)
    W.build()
    W.solve()

    new_emp = Distribution(
        np.array([[0.0, 0.0], [1.2, 0.0]], dtype=np.float64),
        np.array([5, 5], dtype=np.int64),
    )
    new_theo = Distribution(
        np.array([[0.6, 0.0], [1.5, 0.0]], dtype=np.float64),
        np.array([5, 5], dtype=np.int64),
    )
    emp_grad, theo_grads = W.update_positions_and_get_gradient(new_emp, [new_theo])
    # Gradient must be finite and the right shape.  We don't pin the values —
    # that's the job of test_position_gradient.py for the trash-free case;
    # here we just want to confirm combined trash doesn't break the path.
    assert emp_grad.shape == (2, 2)
    assert len(theo_grads) == 1
    assert theo_grads[0].shape == (2, 2)
    assert np.all(np.isfinite(emp_grad))
    assert np.all(np.isfinite(theo_grads[0]))


# ---------------------------------------------------------------------------
# Combined trash isolated/dead-end peak handling
# ---------------------------------------------------------------------------


def test_isolated_emp_uses_cheaper_trash():
    # Two isolated emp@0 (far from any theo) → must use cheapest emp-side trash.
    # max_distance=10 isolates emp@0 from theo@1000.  Combine simple=100 and
    # exp=5; isolated unit should cost 5 each (5 * 2 = 10).
    # Reachable component: emp@1000 (1 unit) ↔ theo@1000 (1 unit) matches at 0.
    emp = _dist1d([0, 1000], [2, 1])
    theo = _dist1d([1000], [1])
    W = WassersteinNetwork(emp, [theo], DistanceMetric.L1, 10)
    W.add_simple_trash(100)
    W.add_experimental_trash(5)
    assert _solve(W) == 2 * 5


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
