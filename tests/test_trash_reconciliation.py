"""
Brute-force cross-checks for the global trash reconciliation.

wnet always splits the network into independent subgraphs and reconciles the
trash cost globally, so the decomposed result must equal the *single-graph*
min-cost flow.  Here we build that single global graph explicitly and solve it
with an independent solver (networkx.network_simplex), then assert wnet agrees,
across:
  - every trash subset (simple / exp / theo and combinations),
  - balanced and unbalanced totals,
  - multi-cluster topologies (forcing decomposition) and isolated nodes,
  - the dense and 1D-chain factories.

We also check signal_part_derivatives against finite differences of total_cost.
"""

import itertools

import numpy as np
import pytest

networkx = pytest.importorskip("networkx")

from wnet import Distribution, WassersteinNetwork
from wnet.distances import DistanceMetric


# --- independent oracle: single-graph min-cost flow of the same model --------


def _l1(col_a, col_b):
    return int(np.abs(col_a - col_b).sum())


def _oracle_cost(emp_pos, emp_int, theo_pos, theo_int, max_dist,
                 c_s=None, c_exp=None, c_theo=None):
    """Single global-graph min-cost flow (the value the decomposition must match).

    Source S pushes F = supply units to sink K, where the supply rule mirrors
    wnet: max(E,T) when both sides have an escape, else the escapable side only.
    Edges: S->emp (cap E_i), emp->theo (cap big, weight=dist) for dist < max_dist,
    theo->K (cap T_j), and the optional trash edges S->K (c_s), emp->K (c_exp),
    S->theo (c_theo).
    """
    E = int(emp_int.sum())
    T = int(theo_int.sum())
    emp_disch = (c_exp is not None) or (c_s is not None)
    theo_fill = (c_theo is not None) or (c_s is not None)
    assert emp_disch or theo_fill, "no trash present"
    if emp_disch and theo_fill:
        F = max(E, T)
    elif emp_disch:
        F = E
    else:
        F = T
    big = F + 1

    G = networkx.DiGraph()
    G.add_node("S", demand=-F)
    G.add_node("K", demand=F)
    ne = emp_pos.shape[1]
    nt = theo_pos.shape[1]
    for i in range(ne):
        G.add_edge("S", ("e", i), capacity=int(emp_int[i]), weight=0)
        if c_exp is not None:
            G.add_edge(("e", i), "K", capacity=big, weight=int(c_exp))
    for j in range(nt):
        G.add_edge(("t", j), "K", capacity=int(theo_int[j]), weight=0)
        if c_theo is not None:
            G.add_edge("S", ("t", j), capacity=big, weight=int(c_theo))
    for i in range(ne):
        for j in range(nt):
            d = _l1(emp_pos[:, i], theo_pos[:, j])
            if d < max_dist:
                G.add_edge(("e", i), ("t", j), capacity=big, weight=d)
    if c_s is not None:
        G.add_edge("S", "K", capacity=big, weight=int(c_s))

    cost, _ = networkx.network_simplex(G)
    return cost


def _wnet_cost(emp_pos, emp_int, theo_pos, theo_int, max_dist,
               c_s=None, c_exp=None, c_theo=None, force_dense_1d=False):
    emp = Distribution(emp_pos.astype(np.float64), emp_int.astype(np.int64))
    theo = Distribution(theo_pos.astype(np.float64), theo_int.astype(np.int64))
    W = WassersteinNetwork(
        emp, [theo], DistanceMetric.L1, max_dist, force_dense_1d=force_dense_1d
    )
    if c_s is not None:
        W.add_simple_trash(c_s)
    if c_exp is not None:
        W.add_experimental_trash(c_exp)
    if c_theo is not None:
        W.add_theoretical_trash(c_theo)
    W.build()
    W.solve()
    return W.total_cost()


# All non-empty subsets of the three trash types (cost values chosen distinct
# so the cheapest-route logic is actually exercised).
_TRASH_SUBSETS = [
    {"c_s": 10},
    {"c_exp": 10},
    {"c_theo": 10},
    {"c_exp": 7, "c_theo": 13},
    {"c_s": 10, "c_exp": 6},
    {"c_s": 10, "c_theo": 8},
    {"c_s": 12, "c_exp": 6, "c_theo": 9},
]


def _gen_clustered_1d(rng, n_clusters=3):
    """Several well-separated 1D clusters → decomposition + possible isolated nodes."""
    emp_pos, emp_int, theo_pos, theo_int = [], [], [], []
    for c in range(n_clusters):
        center = c * 1000
        ne = int(rng.integers(0, 4))
        nt = int(rng.integers(0, 4))
        if ne == 0 and nt == 0:
            ne = 1
        for _ in range(ne):
            emp_pos.append(center + int(rng.integers(0, 5)))
            emp_int.append(int(rng.integers(1, 8)))
        for _ in range(nt):
            theo_pos.append(center + int(rng.integers(0, 5)))
            theo_int.append(int(rng.integers(1, 8)))
    if not emp_pos:
        emp_pos, emp_int = [0], [1]
    if not theo_pos:
        theo_pos, theo_int = [3000], [1]
    return (
        np.array([emp_pos], dtype=float),
        np.array(emp_int),
        np.array([theo_pos], dtype=float),
        np.array(theo_int),
    )


@pytest.mark.parametrize("seed", range(25))
@pytest.mark.parametrize("trash", _TRASH_SUBSETS)
def test_total_cost_matches_single_graph_1d(seed, trash):
    rng = np.random.default_rng(seed)
    emp_pos, emp_int, theo_pos, theo_int = _gen_clustered_1d(rng)
    max_dist = 20  # within-cluster (<5) matchable; between-cluster (1000) not.
    expected = _oracle_cost(emp_pos, emp_int, theo_pos, theo_int, max_dist, **trash)
    # Both factories must reproduce the single-graph value.
    chain = _wnet_cost(emp_pos, emp_int, theo_pos, theo_int, max_dist,
                       force_dense_1d=False, **trash)
    dense = _wnet_cost(emp_pos, emp_int, theo_pos, theo_int, max_dist,
                       force_dense_1d=True, **trash)
    assert chain == expected, (trash, seed, chain, expected)
    assert dense == expected, (trash, seed, dense, expected)


def _gen_clustered_2d(rng, n_clusters=3):
    emp_pos, emp_int, theo_pos, theo_int = [], [], [], []
    for c in range(n_clusters):
        cx, cy = c * 1000, c * 1000
        ne = int(rng.integers(0, 4))
        nt = int(rng.integers(0, 4))
        if ne == 0 and nt == 0:
            ne = 1
        for _ in range(ne):
            emp_pos.append((cx + int(rng.integers(0, 4)), cy + int(rng.integers(0, 4))))
            emp_int.append(int(rng.integers(1, 8)))
        for _ in range(nt):
            theo_pos.append((cx + int(rng.integers(0, 4)), cy + int(rng.integers(0, 4))))
            theo_int.append(int(rng.integers(1, 8)))
    if not emp_pos:
        emp_pos, emp_int = [(0, 0)], [1]
    if not theo_pos:
        theo_pos, theo_int = [(3000, 3000)], [1]
    return (
        np.array(emp_pos, dtype=float).T,
        np.array(emp_int),
        np.array(theo_pos, dtype=float).T,
        np.array(theo_int),
    )


@pytest.mark.parametrize("seed", range(25))
@pytest.mark.parametrize("trash", _TRASH_SUBSETS)
def test_total_cost_matches_single_graph_2d(seed, trash):
    rng = np.random.default_rng(1000 + seed)
    emp_pos, emp_int, theo_pos, theo_int = _gen_clustered_2d(rng)
    max_dist = 20
    expected = _oracle_cost(emp_pos, emp_int, theo_pos, theo_int, max_dist, **trash)
    got = _wnet_cost(emp_pos, emp_int, theo_pos, theo_int, max_dist, **trash)
    assert got == expected, (trash, seed, got, expected)


@pytest.mark.parametrize("seed", range(15))
@pytest.mark.parametrize("trash", _TRASH_SUBSETS)
def test_signal_part_derivatives_finite_difference(seed, trash):
    """signal_part_derivatives must equal total_cost(theo_j += 1) - total_cost()."""
    rng = np.random.default_rng(5000 + seed)
    emp_pos, emp_int, theo_pos, theo_int = _gen_clustered_1d(rng)
    max_dist = 20

    emp = Distribution(emp_pos.astype(np.float64), emp_int.astype(np.int64))
    theo = Distribution(theo_pos.astype(np.float64), theo_int.astype(np.int64))
    W = WassersteinNetwork(emp, [theo], DistanceMetric.L1, max_dist, force_dense_1d=True)
    for k, v in trash.items():
        getattr(W, {"c_s": "add_simple_trash", "c_exp": "add_experimental_trash",
                    "c_theo": "add_theoretical_trash"}[k])(v)
    W.build()
    W.solve()
    base = W.total_cost()
    derivs = W.signal_part_derivatives().get(0, {})

    for j in range(theo_pos.shape[1]):
        bumped = theo_int.copy()
        bumped[j] += 1
        fd = _wnet_cost(emp_pos, emp_int, theo_pos, bumped, max_dist, **trash) - base
        assert derivs[j] == fd, (trash, seed, j, derivs[j], fd)
