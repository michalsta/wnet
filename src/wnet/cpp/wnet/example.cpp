#include "distribution.hpp"
#include "decompositable_graph.hpp"
#include "graph_elements.hpp"
#include "distances.hpp"


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

    WassersteinNetwork<LEMON_INT, LEMON_INT> wnet(
        &dist,
        {&dist2, &dist3},
        l1_distance<2, double>,
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