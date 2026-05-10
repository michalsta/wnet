import numpy as np

from wnet import Distribution, WassersteinNetwork
from wnet.distances import DistanceMetric


def compare(E, T, trash_cost, fractions=None):
    if fractions is None:
        fractions = [1.0] * len(T)

    # val1 = solver.run(fractions)
    # positions = np.concatenate([s.positions for s in T], axis=1)
    # intensities = np.concatenate([s.intensities * f for s, f in zip(T, fractions)])

    decomp_solver = WassersteinNetwork(
        E,
        T,
        DistanceMetric.L2,
        trash_cost,
    )
    # decomp_solver.show()
    # decomp_solver.show_cgraph()
    decomp_solver.add_simple_trash(trash_cost)
    decomp_solver.build()
    decomp_solver.solve(fractions)
    return decomp_solver.total_cost()


def test_compare_1():
    # E at (0,0), T at (1,0): one unit matched at L2 dist=1.
    S1 = Distribution(np.array([[0], [0]]), np.array([1]))
    S2 = Distribution(np.array([[1], [0]]), np.array([1]))
    assert compare(S1, [S2], 10) == 1


def test_compare_2():
    # E=1 unit vs T1+T2=2 units. E matches T2@(1,0) at cost 1; T3@(2,0) trashed at 10.
    S1 = Distribution(np.array([[0], [0]]), np.array([1]))
    S2 = Distribution(np.array([[1], [0]]), np.array([1]))
    S3 = Distribution(np.array([[2], [0]]), np.array([1]))
    assert compare(S1, [S2, S3], 10) == 11


def test_compare_3():
    # E=1 unit vs T1+T2+T3=3 units. E matches T2@(1,0) at cost 1; T3 and T4 trashed at 10 each.
    S1 = Distribution(np.array([[0], [0]]), np.array([1]))
    S2 = Distribution(np.array([[1], [0]]), np.array([1]))
    S3 = Distribution(np.array([[2], [0]]), np.array([1]))
    S4 = Distribution(np.array([[3], [0]]), np.array([1]))
    assert compare(S1, [S2, S3, S4], 10) == 21


"""
def test_compare_4():
    S1 = Spectrum(np.random.randint(0, 1000, (2,5)), np.random.randint(0, 1000, 5))
    S2 = Spectrum(np.random.randint(0, 1000, (2,5)), np.random.randint(0, 1000, 5))

    print(compare(S1, [S2], 10, [1.0]))


def test_compare_5():
    S1 = Spectrum(np.random.randint(0, 1000, (2,50)), np.random.randint(0, 1000, 50))
    S2 = Spectrum(np.random.randint(0, 1000, (2,50)), np.random.randint(0, 1000, 50))
    S3 = Spectrum(np.random.randint(0, 1000, (2,50)), np.random.randint(0, 1000, 50))

    print(compare(S1, [S2, S3], 10, [0.0, 1.0]))

def test_compare_6():
    S1 = Spectrum(np.random.randint(0, 1000, (2,50)), np.random.randint(0, 1000, 50))
    S2 = Spectrum(np.random.randint(0, 1000, (2,50)), np.random.randint(0, 1000, 50))
    S3 = Spectrum(np.random.randint(0, 1000, (2,50)), np.random.randint(0, 1000, 50))
    S4 = Spectrum(np.random.randint(0, 1000, (2,50)), np.random.randint(0, 1000, 50))

    print(compare(S1, [S2, S3, S4], 10, [0.0, 1.0, 1.0]))
"""
if __name__ == "__main__":
    test_compare_1()
    test_compare_2()
    test_compare_3()
    # test_compare_4()
    # test_compare_5()
    # test_compare_6()
