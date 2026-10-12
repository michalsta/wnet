// C++20, -Isrc/wnet/cpp/wnet; also run with software int128 and UBSan.
#include "convex_sweep_repair.hpp"
#include <cassert>
#include <random>

using I = int64_t;
using W = wide_int::wide_t;

template <class Profit>
I assignment_oracle(const std::vector<double> &empirical, const std::vector<double> &theoretical,
                    Profit profit) {
    std::vector<I> dp(size_t(1) << theoretical.size(), -1000000);
    dp[0] = 0;
    for (double x : empirical) {
        auto next = dp;
        for (size_t mask = 0; mask < dp.size(); ++mask)
            for (size_t j = 0; j < theoretical.size(); ++j)
                if (!(mask & (size_t(1) << j)))
                    next[mask | (size_t(1) << j)] = std::max(next[mask | (size_t(1) << j)],
                                                             dp[mask] + profit(x, theoretical[j]));
        dp = std::move(next);
    }
    return *std::max_element(dp.begin(), dp.end());
}

int main() {
    const W above_i64 = W(std::numeric_limits<I>::max()) + W(1);
    assert(W(std::numeric_limits<I>::max()) < above_i64);
    assert(-above_i64 < W(-1));
    assert(-above_i64 - W(1) < -above_i64);

    // A finite residual capacity may equal the infinite-arc sentinel.
    std::vector<double> max_pos{0};
    std::vector<I> max_mass{std::numeric_limits<I>::max()};
    auto zero_cost = [](double, double) -> I { return 0; };
    auto max_result =
        convex_sweep_dual::repair<I>(max_pos, max_mass, max_mass, 1, 2, zero_cost, {});
    assert(std::get<1>(max_result) == W(std::numeric_limits<I>::max()));

    // Wide primal/dual objectives must not be narrowed during correction.
    const I mass = 10000000000LL, tau = 1000000000;
    std::vector<double> positions{0, 1, 2};
    std::vector<I> e{mass, 0, mass}, t{0, mass, mass};
    auto squared = [](double a, double b) -> I { return I((a - b) * (a - b)); };
    std::vector<convex_sweep::Match> initial{{0, 1, mass}, {2, 2, mass}};
    auto [matches, profit, alpha, beta] =
        convex_sweep_dual::repair<I>(positions, e, t, tau, 40000, squared, initial);
    assert(profit == W(2 * tau - 1) * mass);

    std::mt19937 rng(941);
    for (size_t trial = 0; trial < 4000; ++trial) {
        // Small integer capacities expand only for the independent exhaustive
        // assignment oracle. Production correction transfers weighted chunks.
        constexpr size_t n = 4;
        std::vector<double> pos(n), empirical, theoretical;
        std::vector<I> emp(n), theo(n);
        std::vector<unsigned char> possible(n, 1);
        std::vector<convex_sweep::Event> events;
        for (size_t i = 0; i < n; ++i) {
            pos[i] = (i ? pos[i - 1] : 0) + double(rng() % 1001) / 1000;
            emp[i] = rng() % 3;
            theo[i] = rng() % 3;
            if (emp[i])
                events.push_back({pos[i], 0, emp[i], i});
            if (theo[i])
                events.push_back({pos[i], 1, theo[i], i});
            for (I k = 0; k < emp[i]; ++k)
                empirical.push_back(pos[i]);
            for (I k = 0; k < theo[i]; ++k)
                theoretical.push_back(pos[i]);
        }
        const I scale = 1 + rng() % 25, reward = 1 + rng() % 20;
        const double p = 1 + double(rng() % 3) / 2;
        const double radius = std::pow(double(reward + 1) / scale, 1 / p);
        auto cost = [=](double a, double b) -> I {
            return std::llround(scale * std::pow(std::abs(a - b), p));
        };
        auto benefit = [&](double a, double b) { return reward - cost(a, b); };
        const auto trace = convex_sweep::sweep_solve<true>(events, benefit, radius);
        const I best = assignment_oracle(empirical, theoretical, benefit);
        if (convex_sweep_dual::is_monge<I>(pos, emp, possible, reward, radius, cost))
            assert(trace.profit == best);
        auto [fixed, value, a, b] =
            convex_sweep_dual::repair<I>(pos, emp, theo, reward, radius, cost, trace.matches);
        assert(value == W(best)); // repair itself also validates exact dual equality
        const auto gains =
            convex_sweep_dual::repaired_gains<I>(pos, emp, theo, reward, radius, cost, fixed, a, b);
        for (size_t q = 0; q < n; ++q) {
            auto probe = theoretical;
            probe.push_back(pos[q]);
            assert(gains[q] == assignment_oracle(empirical, probe, benefit) - best);
        }
    }
}
