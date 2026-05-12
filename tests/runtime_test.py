from common_testing import create_large_wsdflow_instance, solve_large_wsdflow_instance
import numpy as np

if __name__ == "__main__":
    # params for a middle-sized test
    params = {
        "seed": 123,
        "dimension": 4,
        "pos_range": 200,
        "int_range": 20,
        "size": 500,
        "no_theoretical_distributions": 4,
        "point": (0.2, 0.3, 0.4, 0.5),
    }

    wsd_instance = create_large_wsdflow_instance(**params)[0]

    # We'll run 1000 tests with different points, keeping the instance fixed
    num_tests = 1000
    rng = np.random.default_rng(seed=42)
    test_points = rng.uniform(
        0, 1, size=(num_tests, params["no_theoretical_distributions"])
    )

    # Start timer
    import time

    start_time = time.time()

    total_result = 0
    for i in range(num_tests):
        point = tuple(test_points[i])
        wsd_instance.solve(point)
        total_result += solve_large_wsdflow_instance((wsd_instance, point))
        # print(f"Test {i+1}/{num_tests}, Point: {point}, Result: {total_result}")

    # End timer
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time for {num_tests} tests: {total_time:.2f} seconds")
    print(f"Average time per test: {total_time / num_tests:.4f} seconds")
    print(f"Accumulated result over all tests: {total_result}")
    assert total_result == 120046034
