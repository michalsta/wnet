import numpy as np
import pytest
from wnet import WassersteinNetwork
from wnet.distribution import Distribution_1D
from wnet.distances import DistanceMetric


def make_network_and_solve(base_positions, base_intensities, target_positions, target_intensities, max_distance):
    """Build a truncated Wasserstein network with simple trash, solve it, and return the network."""
    base = Distribution_1D(np.array(base_positions, dtype=np.float64), np.array(base_intensities, dtype=np.int64))
    target = Distribution_1D(np.array(target_positions, dtype=np.float64), np.array(target_intensities, dtype=np.int64))
    W = WassersteinNetwork(base, [target], distance=DistanceMetric.L1, max_distance=int(max_distance))
    W.add_simple_trash(int(max_distance))
    W.build()
    W.solve()
    return W


def perturb_and_solve(base_positions, base_intensities, target_positions, target_intensities, max_distance, signal_index, delta):
    """Rebuild the network with one target signal increased by delta, return total cost."""
    new_intensities = np.array(target_intensities, dtype=np.int64).copy()
    new_intensities[signal_index] += delta
    base = Distribution_1D(np.array(base_positions, dtype=np.float64), np.array(base_intensities, dtype=np.int64))
    target = Distribution_1D(np.array(target_positions, dtype=np.float64), new_intensities)
    W = WassersteinNetwork(base, [target], distance=DistanceMetric.L1, max_distance=int(max_distance))
    W.add_simple_trash(int(max_distance))
    W.build()
    W.solve()
    return W.total_cost()


def test_exact_match_derivative_is_trash_cost():
    """
    Base: pos=0, intensity=5.  Target: pos=10, intensity=5.  trash=100.
    All 5 units match at cost 10. Total cost = 50.
    Adding 1 to target: 5 still match at 10, 1 extra goes to trash at 100.
    New cost = 150. Change = +100 = trash cost.
    """
    W = make_network_and_solve([0], [5], [10], [5], 100)
    assert W.total_cost() == 50

    sg = W.subgraphs()[0]
    derivs = sg.signal_part_derivatives()
    assert derivs[0][0] == 100  # trash cost

    # verify by perturbation
    assert perturb_and_solve([0], [5], [10], [5], 100, 0, 1) - 50 == 100


def test_excess_base_derivative_saves_trash():
    """
    Base: pos=0, intensity=10.  Target: pos=10, intensity=5.  trash=100.
    5 match at cost 10, 5 excess go to trash at 100. Total = 550.
    Adding 1 to target: 6 match at 10, 4 to trash. New = 460. Change = -90 = 10 - 100.
    The extra target unit absorbs one unit from trash (saves 100) but pays distance 10.
    """
    W = make_network_and_solve([0], [10], [10], [5], 100)
    assert W.total_cost() == 550

    sg = W.subgraphs()[0]
    derivs = sg.signal_part_derivatives()
    assert derivs[0][0] == -90  # distance - trash = 10 - 100

    assert perturb_and_solve([0], [10], [10], [5], 100, 0, 1) - 550 == -90


def test_two_targets_different_distances():
    """
    Base: pos=0, intensity=10.
    Target 0: pos=5, intensity=3.  Target 1: pos=20, intensity=3.  trash=100.
    3 match target0 at 5, 3 match target1 at 20, 4 to trash at 100. Total = 475.
    Increase target0 by 1: saves a trash unit, pays distance 5. Change = 5 - 100 = -95.
    Increase target1 by 1: saves a trash unit, pays distance 20. Change = 20 - 100 = -80.
    """
    W = make_network_and_solve([0], [10], [5, 20], [3, 3], 100)
    assert W.total_cost() == 475

    sg = W.subgraphs()[0]
    derivs = sg.signal_part_derivatives()
    assert derivs[0][0] == -95  # 5 - 100
    assert derivs[0][1] == -80  # 20 - 100

    assert perturb_and_solve([0], [10], [5, 20], [3, 3], 100, 0, 1) - 475 == -95
    assert perturb_and_solve([0], [10], [5, 20], [3, 3], 100, 1, 1) - 475 == -80


def test_two_bases_nearest_reroutes():
    """
    Base 0: pos=0, intensity=5.  Base 1: pos=100, intensity=5.
    Target: pos=10, intensity=5.  trash=200.
    All 5 target units come from base0 at cost 10. Base1's 5 go to trash at 200.
    Total = 50 + 1000 = 1050.
    Increase target by 1: base0 full (5 cap), so extra unit from base1 at cost 90, saves 1 trash.
    New = 50 + 90 + 800 = 940. Change = -110 = 90 - 200.
    """
    W = make_network_and_solve([0, 100], [5, 5], [10], [5], 200)
    assert W.total_cost() == 1050

    sg = W.subgraphs()[0]
    derivs = sg.signal_part_derivatives()
    assert derivs[0][0] == -110  # 90 - 200

    assert perturb_and_solve([0, 100], [5, 5], [10], [5], 200, 0, 1) - 1050 == -110


def test_excess_theo_derivative_is_trash_cost():
    """
    Base: pos=0, intensity=3.  Target: pos=10, intensity=5.  trash=100.
    emp(3) < theo(5): total_flow = 5. 3 match at 10, 2 to trash at 100.
    Total = 30 + 200 = 230.
    Adding 1 to target: total_flow = 6. 3 match, 3 to trash. New = 330.
    Change = +100 = trash_cost (extra unit just goes to trash).
    """
    W = make_network_and_solve([0], [3], [10], [5], 100)
    assert W.total_cost() == 230

    sg = W.subgraphs()[0]
    derivs = sg.signal_part_derivatives()
    assert derivs[0][0] == 100

    assert perturb_and_solve([0], [3], [10], [5], 100, 0, 1) - 230 == 100


def test_excess_theo_with_slack_cheap_node():
    """
    Base: pos=0, intensity=3.
    Target 0: pos=5, intensity=3.  Target 1: pos=50, intensity=3.  trash=100.
    emp(3) < theo(6): total_flow = 6. 3 match target0 at 5, 3 to trash.
    Target1 has full slack (flow=0, cap=3).
    Total = 15 + 300 = 315.
    Increase target0 by 1: total_flow = 7. 3 match t0, 4 to trash. Change = +100.
    Increase target1 by 1: total_flow = 7. Still 3→t0, 4 to trash. Change = +100.
    Despite slack on target1, derivative is trash_cost (not 0).
    """
    W = make_network_and_solve([0], [3], [5, 50], [3, 3], 100)
    assert W.total_cost() == 315

    sg = W.subgraphs()[0]
    derivs = sg.signal_part_derivatives()
    assert derivs[0][0] == 100
    assert derivs[0][1] == 100

    assert perturb_and_solve([0], [3], [5, 50], [3, 3], 100, 0, 1) - 315 == 100
    assert perturb_and_solve([0], [3], [5, 50], [3, 3], 100, 1, 1) - 315 == 100


def test_excess_theo_slack_node_cheaper_than_trash():
    """
    Base: pos=0, intensity=5.
    Target 0: pos=10, intensity=5.  Target 1: pos=20, intensity=5.  trash=100.
    emp(5) < theo(10): total_flow = 10. All 5 base units go to target0 at 10.
    target1 has full slack. 5 units go to trash.
    Total = 50 + 500 = 550.
    Increase target1 by 1: total_flow = 11. Extra supply unit could:
      - go to trash (cost 100)
      - match to target1 via source→base→target1 (cost 20)
        but this adds 1 supply → 1 more trash (+100), net = 20 + 100 = 120? No.
        Actually src_adjust=0 when emp≤theo, so via-source cost = dist_src + 0.
        But we also need sink_adjust for reroute...
    Let's just verify by perturbation.
    """
    W = make_network_and_solve([0], [5], [10, 20], [5, 5], 100)
    original = W.total_cost()

    sg = W.subgraphs()[0]
    derivs = sg.signal_part_derivatives()

    for i in range(2):
        new_cost = perturb_and_solve([0], [5], [10, 20], [5, 5], 100, i, 1)
        assert derivs[0][i] == new_cost - original, (
            f"peak {i}: predicted={derivs[0][i]}, actual={new_cost - original}"
        )


def test_excess_base_slack_node_derivative_zero():
    """
    Base: pos=0, intensity=10.
    Target 0: pos=5, intensity=3.  Target 1: pos=90, intensity=1.  trash=100.
    emp(10) > theo(4). All 3 match target0 at 5, 1 matches target1 at 90.
    6 excess to trash. Total = 15 + 90 + 600 = 705.
    Increase target0 by 1: saves trash, pays 5. Change = 5 - 100 = -95.
    """
    W = make_network_and_solve([0], [10], [5, 90], [3, 1], 100)
    original = W.total_cost()

    sg = W.subgraphs()[0]
    derivs = sg.signal_part_derivatives()

    for i in range(2):
        new_cost = perturb_and_solve([0], [10], [5, 90], [3, 1], 100, i, 1)
        assert derivs[0][i] == new_cost - original, (
            f"peak {i}: predicted={derivs[0][i]}, actual={new_cost - original}"
        )


def test_equal_supply_demand_multiple_nodes():
    """
    Balanced case with multiple bases and targets.
    emp == theo, so no trash flow. Each +1 increases total_flow.
    """
    W = make_network_and_solve([0, 50], [5, 5], [10, 40], [5, 5], 100)
    original = W.total_cost()

    sg = W.subgraphs()[0]
    derivs = sg.signal_part_derivatives()

    for i in range(2):
        new_cost = perturb_and_solve([0, 50], [5, 5], [10, 40], [5, 5], 100, i, 1)
        assert derivs[0][i] == new_cost - original, (
            f"peak {i}: predicted={derivs[0][i]}, actual={new_cost - original}"
        )


def test_single_unit_intensities():
    """Minimal intensities (1 each) — boundary case."""
    W = make_network_and_solve([0], [1], [10], [1], 100)
    original = W.total_cost()

    sg = W.subgraphs()[0]
    derivs = sg.signal_part_derivatives()

    new_cost = perturb_and_solve([0], [1], [10], [1], 100, 0, 1)
    assert derivs[0][0] == new_cost - original


def test_many_targets_one_base():
    """One base feeding many targets — tests rerouting among many nodes."""
    positions = [10, 20, 30, 40, 50]
    intensities = [2, 2, 2, 2, 2]
    W = make_network_and_solve([0], [15], positions, intensities, 100)
    original = W.total_cost()

    sg = W.subgraphs()[0]
    derivs = sg.signal_part_derivatives()

    for i in range(5):
        new_cost = perturb_and_solve([0], [15], positions, intensities, 100, i, 1)
        assert derivs[0][i] == new_cost - original, (
            f"peak {i}: predicted={derivs[0][i]}, actual={new_cost - original}"
        )


def test_many_bases_one_target():
    """Many bases competing for one target — tests source-path selection."""
    base_pos = [0, 20, 40, 60, 80]
    base_int = [3, 3, 3, 3, 3]
    W = make_network_and_solve(base_pos, base_int, [50], [5], 100)
    original = W.total_cost()

    sg = W.subgraphs()[0]
    derivs = sg.signal_part_derivatives()

    new_cost = perturb_and_solve(base_pos, base_int, [50], [5], 100, 0, 1)
    assert derivs[0][0] == new_cost - original


@pytest.mark.parametrize("seed", range(50))
def test_signal_part_derivatives_predict_cost_change(seed):
    """
    For each signal in the target distribution, increase it by 1 and verify that
    the actual change in total cost matches the derivative prediction.

    The derivative of total cost w.r.t. increasing signal i by 1 should equal
    the shortest-path distance from source to that signal's node in the residual graph.
    """
    rng = np.random.default_rng(seed)

    n_base = rng.integers(1, 10)
    n_target = rng.integers(1, 10)

    base_positions = (rng.uniform(0, 100, size=n_base) * 1000).astype(np.int64)
    base_intensities = rng.integers(1, 20, size=n_base)
    target_positions = (rng.uniform(0, 100, size=n_target) * 1000).astype(np.int64)
    target_intensities = rng.integers(1, 20, size=n_target)

    max_distance = 50000

    W = make_network_and_solve(base_positions, base_intensities, target_positions, target_intensities, max_distance)
    original_cost = W.total_cost()

    # Collect derivatives across all subgraphs
    derivatives = {}
    for sg in W.subgraphs():
        for spec_id, peaks in sg.signal_part_derivatives().items():
            derivatives.setdefault(spec_id, {}).update(peaks)
    # We have one target distribution (spectrum_id=0)
    assert 0 in derivatives

    delta = 1
    for peak_idx in range(n_target):
        if peak_idx not in derivatives.get(0, {}):
            # Peak is disconnected (no edges within max_distance) — not
            # part of any subgraph, so no derivative to test.
            continue
        predicted_derivative = derivatives[0][peak_idx]
        new_cost = perturb_and_solve(
            base_positions, base_intensities,
            target_positions, target_intensities,
            max_distance, peak_idx, delta,
        )
        actual_change = new_cost - original_cost
        assert actual_change == predicted_derivative * delta, (
            f"seed={seed}, peak_idx={peak_idx}: "
            f"predicted={predicted_derivative * delta}, actual={actual_change}, "
            f"original_cost={original_cost}, new_cost={new_cost}"
        )
