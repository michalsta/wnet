#include "distribution.hpp"
#include "decompositable_graph.hpp"
#include "graph_elements.hpp"


int main()
{
    auto rng = std::mt19937(42);
    VectorDistribution<2> dist = VectorDistribution<2>::CreateRandom(
        10000,
        100.0,
        10,
        rng
    );
    VectorDistribution<2> dist2 = VectorDistribution<2>::CreateRandom(
        10000,
        100.0,
        10,
        rng
    );
    VectorDistribution<2> dist3 = VectorDistribution<2>::CreateRandom(
        10000,
        100.0,
        10,
        rng
    );

    std::function<LEMON_INT(const VectorDistribution<2>::Point_t&,
                            const VectorDistribution<2>::Point_t&)> dist_func =
        [](const VectorDistribution<2>::Point_t& p1,
           const VectorDistribution<2>::Point_t& p2) -> VectorDistribution<2>::intensity_type {
            return static_cast<VectorDistribution<2>::intensity_type>(l1_distance<2, double>(p1, p2));
        };

    WassersteinNetwork<LEMON_INT, VectorDistribution<2>> wnet(
        &dist,
        {&dist2, &dist3},
        dist_func,
        10
    );

    std::cout << "WassersteinNetwork created with "
              << wnet.no_nodes() << " nodes and "
              << wnet.no_edges() << " edges."
              << std::endl;

    wnet.build();
    std::vector<double> point = {0.5, 0.5};
    for (size_t iter = 0; iter < 20; ++iter) {
        wnet.solve(point);
        std::cout << "Iteration " << iter << ", total cost: " << wnet.total_cost() << std::endl;
    }
}