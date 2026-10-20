"""NetworkSimplex warm/cold policy fields, set at build or between solves.

* ``large_repair_mass_fraction`` scopes a violated-mass gate to subgraphs
  above ``repair_max_nodes``: such a subgraph repairs only while the violated
  basic-arc mass is at most that fraction of its positive supply; smaller
  subgraphs keep ``warm_violation_limit``.
* ``warm_violation_mass_fraction`` applies the same gate to every subgraph.
* ``set_warm_policy()`` changes these fields between solves, keeping the
  basis, and refined copies inherit them.
* ``ns_stats()`` reports work units, policy cold starts and the violation
  measurements.

Every policy must reproduce the optimal costs of independent cold solves.
"""
import numpy as np
import pytest

from wnet import Distribution, WassersteinNetwork
from wnet.distances import DistanceMetric
from wnet import wnet_cpp
from wnet.wnet_cpp import NetworkSimplex, WarmMode

if not getattr(wnet_cpp, "HAS_WARM_MASS_GATE", False):
    pytest.skip("pylmcf without the violated-mass repair gate", allow_module_level=True)

WEIGHTS = np.concatenate([np.linspace(0.5, 1.6, 10), 1.6 + np.cumsum(np.full(10, 2e-3))])


def config(**fields):
    cfg = NetworkSimplex()
    cfg.warm = WarmMode.DualRatio
    for k, v in fields.items():
        setattr(cfg, k, v)
    return cfg


def network(cfg, seed=5, n=200, cap=8):
    rng = np.random.default_rng(seed)
    p1, p2 = rng.uniform(0, 40, (2, n)), rng.uniform(0, 40, (2, n))
    net = WassersteinNetwork(Distribution(p1, rng.uniform(0.2, 1, n)),
                             [Distribution(p2, rng.uniform(0.2, 1, n))],
                             DistanceMetric.L1, max_distance=cap, solver=cfg)
    net.add_simple_trash(8)
    net.build()
    return net


def cold_reference(seed=5, n=200, cap=8):
    net = network(config(warm=WarmMode.NONE), seed, n, cap)
    out = []
    for w in WEIGHTS:
        net.solve([w])
        out.append(net.total_cost())
    return np.array(out)


def run(cfg, seed=5, n=200, cap=8, between=None):
    net = network(cfg, seed, n, cap)
    costs, stats = [], []
    for k, w in enumerate(WEIGHTS):
        if between is not None:
            between(net, k)
        before = dict(net.ns_stats())
        r0, c0 = net.wnet.dual_repair_count(), net.wnet.cold_start_count()
        net.solve([w])
        after = dict(net.ns_stats())
        costs.append(net.total_cost())
        stats.append(dict(repairs=net.wnet.dual_repair_count() - r0,
                          colds=net.wnet.cold_start_count() - c0,
                          policy=after["policy_cold"] - before["policy_cold"],
                          work=after["work_units"] - before["work_units"],
                          violations=after["violations"], mass=after["violation_mass"],
                          supply=after["positive_supply"]))
    return np.array(costs), stats, net


def largest(net):
    return max(net.get_subgraph(i).no_nodes() for i in range(net.no_subgraphs()))


def test_defaults_keep_existing_behaviour():
    cfg = NetworkSimplex()
    assert cfg.large_repair_mass_fraction == -1.0
    assert cfg.warm_violation_mass_fraction == -1.0
    assert cfg.warm_repair_budget == -1.0


def test_large_gate_negative_is_the_old_size_gate():
    ref = cold_reference()
    _, _, net = run(config())
    big = largest(net)
    old, s_old, _ = run(config(repair_max_nodes=big - 1))
    new, s_new, _ = run(config(repair_max_nodes=big - 1, large_repair_mass_fraction=-1.0))
    np.testing.assert_allclose(old, ref, rtol=1e-12)
    np.testing.assert_array_equal(old, new)
    assert [s["repairs"] for s in s_old] == [s["repairs"] for s in s_new]


@pytest.mark.parametrize("fraction", [0.0, 1e-3, 1e-2, 0.05, 0.3, 1e300])
def test_large_gate_follows_the_mass_rule(fraction):
    ref = cold_reference()
    _, _, net = run(config())
    big = largest(net)
    costs, stats, net = run(config(repair_max_nodes=big - 1, large_repair_mass_fraction=fraction))
    np.testing.assert_allclose(costs, ref, rtol=1e-12)
    assert net.no_subgraphs() == 1, "fixture must be one large component"
    for k, s in enumerate(stats[1:], 1):
        if s["violations"] <= 0:
            continue                       # tree patch succeeded or no warm path
        skip = s["mass"] > fraction * s["supply"]
        if s["policy"]:
            assert skip, (k, s)
            assert s["repairs"] == 0
    if fraction == 0.0:
        assert sum(s["repairs"] for s in stats) == 0
    if fraction == 1e300:
        assert sum(s["policy"] for s in stats) == 0
        assert sum(s["repairs"] for s in stats) > 0


def test_large_gate_leaves_small_subgraphs_alone():
    # Many small components, all below repair_max_nodes: a large-only gate of
    # 0 must not suppress their repairs, while the global gate does.
    ref = cold_reference(seed=9, n=300, cap=2)
    free, s_free, net = run(config(), seed=9, n=300, cap=2)
    assert net.no_subgraphs() > 20
    small = largest(net) + 1
    scoped, s_scoped, _ = run(config(repair_max_nodes=small, large_repair_mass_fraction=0.0),
                              seed=9, n=300, cap=2)
    global_, s_global, _ = run(config(warm_violation_mass_fraction=0.0), seed=9, n=300, cap=2)
    for c in (free, scoped, global_):
        np.testing.assert_allclose(c, ref, rtol=1e-12)
    assert sum(s["repairs"] for s in s_free) > 0
    assert [s["repairs"] for s in s_scoped] == [s["repairs"] for s in s_free]
    assert sum(s["repairs"] for s in s_global) == 0


def test_policy_changes_between_solves_keep_the_basis():
    ref = cold_reference()
    _, _, net = run(config())
    big = largest(net)
    never = config(repair_max_nodes=big - 1)
    always = config(repair_max_nodes=-1, warm_violation_limit=-1)

    def alternate(net, k):
        net.set_warm_policy(never if k % 2 else always)

    costs, stats, _ = run(config(), between=alternate)
    np.testing.assert_allclose(costs, ref, rtol=1e-12)
    odd = [s for k, s in enumerate(stats) if k % 2 and k > 0]
    assert sum(s["repairs"] for s in odd) == 0
    assert sum(s["repairs"] for k, s in enumerate(stats) if k % 2 == 0 and k > 0) > 0
    # The basis survives the policy change: warm solves stay cheaper than a
    # full cold start on the small late steps.
    assert stats[-1]["work"] < stats[0]["work"]


def test_refined_copy_inherits_a_runtime_policy():
    net = network(config())
    net.solve([1.0])
    net.set_warm_policy(config(repair_max_nodes=7, large_repair_mass_fraction=0.25,
                               warm_violation_mass_fraction=0.5, warm_repair_budget=3.0))
    refined = net.wnet.refined_copy(net.intensity_scale_factor() * 10)
    # Behavioural check through the refined copy: one large component with a
    # zero gate never repairs.
    net.set_warm_policy(config(repair_max_nodes=1, large_repair_mass_fraction=0.0))
    refined = net.wnet.refined_copy(net.intensity_scale_factor() * 10)
    refined.solve([1.0])
    for w in WEIGHTS:
        refined.solve([w])
    assert refined.dual_repair_count() == 0
    assert dict(refined.ns_stats())["policy_cold"] > 0


def test_ns_stats_counts_work_and_violations():
    _, stats, net = run(config(warm_violation_mass_fraction=1e300))
    # The first solve is a full cold run; a warm solve that the tree patch
    # finishes does no counted simplex work.
    assert stats[0]["work"] > 0
    assert all(s["work"] >= 0 for s in stats)
    measured = [s for s in stats[1:] if s["violations"] > 0]
    assert measured, "no failed tree patch to measure"
    assert all(0 < s["mass"] and s["supply"] > 0 for s in measured)
    total = dict(net.ns_stats())
    assert total["work_units"] == sum(s["work"] for s in stats)
