from wnet import Distribution, WassersteinNetwork
from wnet.distances import wrap_distance_function
import numpy as np


def compare(E, T, trash_cost, fractions=None):
    if fractions is None:
        fractions = [1.0] * len(T)

    #val1 = solver.run(fractions)
    positions = np.concatenate([s.positions for s in T], axis=1)
    intensities = np.concatenate([s.intensities * f for s, f in zip(T, fractions)])

    decomp_solver = WassersteinNetwork(
        E, T, wrap_distance_function(lambda x, y: np.linalg.norm(x - y, axis=0)), trash_cost
    )
    # decomp_solver.show()
    # decomp_solver.show_cgraph()
    decomp_solver.add_simple_trash(trash_cost)
    decomp_solver.build()
    val4 = decomp_solver.solve(fractions)
    #print(
    #    f"Solver: {val1}, Wasserstein: {val2}, Wasserstein_compat: {val3}, DecompositableFlowGraph: {val4}"
    #)
    # assert val1 == val2 # 2 uses diffrent trash so not really the same
    #assert val1 == val3
    #assert val1 == val4
    #return val1, val2, val3, val4
    return val4

def test_compare_1():
    S1 = Distribution(np.array([[0], [0]]), np.array([1]))
    S2 = Distribution(np.array([[1], [0]]), np.array([1]))

    print(compare(S1, [S2], 10))


def test_compare_2():
    S1 = Distribution(np.array([[0], [0]]), np.array([1]))
    S2 = Distribution(np.array([[1], [0]]), np.array([1]))
    S3 = Distribution(np.array([[2], [0]]), np.array([1]))

    print(compare(S1, [S2, S3], 10))


def test_compare_3():
    S1 = Distribution(np.array([[0], [0]]), np.array([1]))
    S2 = Distribution(np.array([[1], [0]]), np.array([1]))
    S3 = Distribution(np.array([[2], [0]]), np.array([1]))
    S4 = Distribution(np.array([[3], [0]]), np.array([1]))

    print(compare(S1, [S2, S3, S4], 10))


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
