// Build with the wnet/cpp/wnet and pylmcf/cpp include paths, C++20,
// -DWNET_FORCE_SOFT_INT128 and -fsanitize=undefined to check the portable
// accumulator and exact primal-dual equality beyond an int64 objective.
#define LEMON_ONLY_TEMPLATES
#include "convex_sweep_dual.hpp"
#include <cassert>

int main() {
    using I = int64_t;
    using W = wide_int::wide_t;
    std::vector<double> positions{0, 1, 2};
    const I mass = 10000000000LL, tau = 1000000000;
    std::vector<I> e{mass, 0, mass}, t{0, mass, mass};
    auto cost = [](double a, double b) -> I { return I((a - b) * (a - b)); };
    const W profit = W(2 * tau - 1) * mass;
    auto [alpha, beta] = convex_sweep_dual::recover<I>(positions, e, t, tau, 40000, cost, profit);
    W objective = 0;
    for (size_t i = 0; i < positions.size(); ++i) {
        assert(alpha[i] >= -tau && alpha[i] <= 0);
        assert(beta[i] >= -tau && beta[i] <= 0);
        objective += W(alpha[i]) * e[i] + W(beta[i]) * t[i];
        for (size_t j = 0; j < positions.size(); ++j)
            assert(alpha[i] <= cost(positions[i], positions[j]) - tau - beta[j]);
    }
    assert(objective == -profit);
    bool rejected = false;
    try {
        convex_sweep_dual::recover<I>(positions, e, t, tau, 40000, cost, profit + W(1));
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    assert(rejected);
}
