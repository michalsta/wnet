import numpy as np
from wnet import Distribution, WassersteinNetwork
from wnet.distances import L1Distance

def create_large_distribution(seed, dimension, pos_range, int_range, size):
    rng = np.random.default_rng(seed)
    data = rng.uniform(size=(dimension, size)) * pos_range
    intensities = rng.uniform(size=(size,)) * int_range
    return Distribution(data, intensities)

def create_large_wsdflow_instance(seed, dimension, pos_range, int_range, size, no_theoretical_distributions, point):
    experimental_distribution = create_large_distribution(seed, dimension, pos_range, int_range, size)
    theoretical_distributions = [
        create_large_distribution(seed + i + 1, dimension, pos_range, int_range, size)
        for i in range(no_theoretical_distributions)
    ]
    wsdflow_instance = WassersteinNetwork(
        experimental_distribution,
        theoretical_distributions,
        distance=L1Distance(),
        max_distance=int(pos_range * dimension / 10),
    )
    wsdflow_instance.build()
    return (wsdflow_instance, point)

def solve_large_wsdflow_instance(wsdflow_instance_point):
    wsdflow_instance, point = wsdflow_instance_point
    if point is not None:
        wsdflow_instance.solve(point)
    else:
        wsdflow_instance.solve()
    return wsdflow_instance.total_cost()

parameter_set = [
    {
        "seed": 42,
        "dimension": 3,
        "pos_range": 100,
        "int_range": 10,
        "size": 1000,
        "no_theoretical_distributions": 5,
        "point": (0.3, 0.4, 10.5),
    },
    {
        "seed": 7,
        "dimension": 2,
        "pos_range": 50,
        "int_range": 5,
        "size": 2000,
        "no_theoretical_distributions": 3,
        "point": (0.1, 0.9),
    },
    {
        "seed": 123,
        "dimension": 4,
        "pos_range": 200,
        "int_range": 20,
        "size": 500,
        "no_theoretical_distributions": 4,
        "point": (0.2, 0.3, 0.4, 0.5),
    },
    # more, bigger ones:
    {
        "seed": 99,
        "dimension": 5,
        "pos_range": 150,
        "int_range": 15,
        "size": 5000,
        "no_theoretical_distributions": 6,
        "point": (0.5, 0.5, 0.5, 0.5, 0.5),
    },
    {
        "seed": 2024,
        "dimension": 6,
        "pos_range": 300,
        "int_range": 30,
        "size": 10000,
        "no_theoretical_distributions": 8,
        "point": None
    },
]

# wsd_instances = [
#     create_large_wsdflow_instance(**params) for params in parameter_set
# ]

expected_results = [21182, 730, 132388, 561612, 5134997]

try:
    import pytest
    @pytest.mark.parametrize("params, expected", zip(
        parameter_set,
        expected_results
    ))
    def test_large_wsdflow_instances(params, expected):
        wsd_instance = create_large_wsdflow_instance(**params)
        cost = solve_large_wsdflow_instance(wsd_instance)
        assert np.isclose(cost, expected)
except ImportError:
    pass

if __name__ == "__main__":
    instances = [create_large_wsdflow_instance(**params) for params in parameter_set]
    costs = []
    for i, instance in enumerate(instances):
        print(f"Solving instance {i + 1}...")
        cost = solve_large_wsdflow_instance(instance)
        print(f"Instance {i + 1}: Total Cost = {cost}")
        costs.append(cost)
    print("All costs:", costs)