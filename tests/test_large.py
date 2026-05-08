from common_testing import create_large_wsdflow_instance, solve_large_wsdflow_instance
import numpy as np

parameter_set = [
    {
        "seed": 42,
        "dimension": 3,
        "pos_range": 100,
        "int_range": 10,
        "size": 1000,
        "no_theoretical_distributions": 5,
        "point": (0.3, 0.4, 10.5, 0.2, 0.6),
    },
    {
        "seed": 7,
        "dimension": 2,
        "pos_range": 50,
        "int_range": 5,
        "size": 2000,
        "no_theoretical_distributions": 3,
        "point": (0.1, 0.9, 0.5),
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
        "point": (0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
    },
    {
        "seed": 2024,
        "dimension": 6,
        "pos_range": 300,
        "int_range": 30,
        "size": 10000,
        "no_theoretical_distributions": 8,
        "point": None,
    },
]

# wsd_instances = [
#     create_large_wsdflow_instance(**params) for params in parameter_set
# ]

expected_results = [1458992, 2792, 229908, 5235286, 188445197]

try:
    import pytest

    @pytest.mark.long
    @pytest.mark.parametrize("params, expected", zip(parameter_set, expected_results))
    def test_large_wsdflow_instances(params, expected):
        wsd_instance = create_large_wsdflow_instance(**params)
        cost = solve_large_wsdflow_instance(wsd_instance)
        print(f"Computed cost: {cost}, Expected cost: {expected}")
        assert cost == expected

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
