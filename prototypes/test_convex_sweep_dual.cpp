// Build with the wnet/cpp/wnet include path, C++20,
// -DWNET_FORCE_SOFT_INT128 and -fsanitize=undefined to check the portable
// accumulator and exact primal-dual equality beyond an int64 objective.
#include "convex_sweep_dual.hpp"
#include <cassert>
#include <random>
#include <iostream>

int main() {
    using I = int64_t;
    using W = wide_int::wide_t;
    std::vector<double> positions{0, 1, 2};
    const I mass = 10000000000LL, tau = 1000000000;
    std::vector<I> e{mass, 0, mass}, t{0, mass, mass};
    auto cost = [](double a, double b) -> I { return I((a - b) * (a - b)); };
    const W profit = W(2 * tau - 1) * mass;
    std::vector<convex_sweep::Match> matching{{0, 1, mass}, {2, 2, mass}};
    auto [alpha, beta] = convex_sweep_dual::recover<I>(positions, e, t, tau, 40000, cost, matching, profit);
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
        convex_sweep_dual::recover<I>(positions, e, t, tau, 40000, cost, matching, profit + W(1));
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    assert(rejected);
    std::mt19937 random(42);
    for (size_t trial = 0; trial < 3000; ++trial) {
        const size_t n = 1 + random() % 12;
        std::vector<double> pos(n);
        std::vector<I> emp(n), theo(n);
        std::vector<convex_sweep::Event> events;
        std::vector<double> eu, tu;
        for (size_t i = 0; i < n; ++i) {
            pos[i] = (i ? pos[i-1] : 0) + random() % 3;
            emp[i] = random() % 5;
            theo[i] = random() % 5;
            if (emp[i]) events.push_back({pos[i], 0, emp[i], i});
            if (theo[i]) events.push_back({pos[i], 1, theo[i], i});
            for (I k = 0; k < emp[i]; ++k) eu.push_back(pos[i]);
            for (I k = 0; k < theo[i]; ++k) tu.push_back(pos[i]);
        }
        const I reward = random() % 20;
        const double radius = std::sqrt(double(reward) + 1);
        auto benefit = [reward, cost](double a, double b) { return reward - cost(a,b); };
        std::vector<I> previous(tu.size()+1, 0), next(previous);
        for (double a : eu) {
            next[0] = 0;
            for (size_t j = 1; j <= tu.size(); ++j)
                next[j] = std::max({next[j-1], previous[j], previous[j-1]+benefit(a,tu[j-1])});
            previous.swap(next);
        }
        const auto cost_only = convex_sweep::sweep_solve<false>(events, benefit, radius);
        static_assert(std::is_same_v<decltype(cost_only), const int64_t>);
        try {
            const auto traced = convex_sweep::sweep_solve<true>(events, benefit, radius);
            assert(cost_only == previous.back());
            assert(traced.profit == cost_only);
            const auto tree_cost = convex_sweep::sweep_solve<false, convex_sweep::TreePLF>(events, benefit, radius);
            const auto tree_trace = convex_sweep::sweep_solve<true, convex_sweep::TreePLF>(events, benefit, radius);
            assert(tree_cost == cost_only && tree_trace.profit == cost_only);
            convex_sweep_dual::recover<I>(pos, emp, theo, reward, radius, cost,
                                          tree_trace.matches, W(tree_trace.profit));
            std::vector<convex_sweep::Snapshot> flat_snapshots;
            std::vector<convex_sweep::BasicSnapshot<convex_sweep::TreePLF>> tree_snapshots;
            assert(convex_sweep::forward_store(events, benefit, radius, flat_snapshots) ==
                   convex_sweep::forward_store(events, benefit, radius, tree_snapshots));
            std::vector<I> flat_marginals, tree_marginals;
            convex_sweep::adjoint_compose(events, benefit, flat_snapshots, flat_marginals);
            convex_sweep::adjoint_compose(events, benefit, tree_snapshots, tree_marginals);
            assert(flat_marginals == tree_marginals);
            convex_sweep_dual::recover<I>(pos, emp, theo, reward, radius, cost,
                                          traced.matches, W(traced.profit));
        } catch (const std::exception& exc) {
            std::cerr << "trial " << trial << ": " << exc.what() << "\n";
            for (const auto& ev : events) std::cerr << ev.pos << ',' << ev.side << ',' << ev.count << ',' << ev.index << ' ';
            std::cerr << " reward=" << reward << '\n';
            throw;
        }
    }
}
