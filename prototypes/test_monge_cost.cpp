#include "wnet/monge_cost.hpp"
#include "wnet/convex_sweep_dual.hpp"
#include <cassert>
#include <cstdio>
#include <random>

using I = int64_t;
using W = wide_int::wide_t;

template<class Profile>
void check(const std::vector<double>& pos, const std::vector<I>& e,
           const std::vector<I>& t, I tau, const monge_cost::GridPower& model) {
    auto cost = [&](double a, double b) { return model.cost(a, b); };
    auto profit = [&](double a, double b) { return tau - cost(a, b); };
    std::vector<size_t> cols;
    std::vector<convex_sweep::Event> events;
    for (size_t i = 0; i < pos.size(); ++i) {
        if (e[i]) events.push_back({pos[i], 0, e[i], i});
        if (t[i]) events.push_back({pos[i], 1, t[i], i});
        for (I k = 0; k < t[i]; ++k) cols.push_back(i);
    }
    // Unrestricted subset assignment oracle: crossing matches are allowed.
    std::vector<I> dp(size_t(1) << cols.size(), -1000000000);
    dp[0] = 0;
    for (size_t i = 0; i < pos.size(); ++i) for (I unit = 0; unit < e[i]; ++unit) {
        auto next = dp;
        for (size_t mask = 0; mask < dp.size(); ++mask)
            for (size_t j = 0; j < cols.size(); ++j) if (!(mask & (size_t(1) << j)))
                next[mask | (size_t(1) << j)] = std::max(next[mask | (size_t(1) << j)],
                    dp[mask] + profit(pos[i], pos[cols[j]]));
        dp.swap(next);
    }
    const I expected = *std::max_element(dp.begin(), dp.end());
    assert((convex_sweep::sweep_solve<false, Profile>(events, profit, model.radius(tau)) == expected));
    auto trace = convex_sweep::sweep_solve<true, Profile>(events, profit, model.radius(tau));
    assert(trace.profit == expected);
    auto [alpha, beta] = convex_sweep_dual::recover<I>(pos, e, t, tau,
        model.radius(tau), cost, trace.matches, W(expected));
    for (size_t i = 0; i < pos.size(); ++i)
        for (size_t j = 0; j < pos.size(); ++j)
            assert(!(W(cost(pos[i], pos[j])) - W(tau) < W(alpha[i]) + W(beta[j])));
}

int main() {
    {
        // Distant costs need not fit int64 when the profitable window does.
        monge_cost::GridPower model;
        std::vector<double> pos;
        model.reset(std::vector<double>{0, .01, 100}, 1000000000000000000LL, 2, pos);
        assert(W(std::numeric_limits<I>::max()) < model.wide_cost(pos[0], pos[2]));
        std::vector<convex_sweep::Event> events{{pos[0], 0, 1, 0}, {pos[1], 1, 1, 1}, {pos[2], 1, 1, 2}};
        const I tau = 1000000000000000LL;
        auto cost = [&](double a, double b) { return model.cost(a, b); };
        auto profit = [&](double a, double b) { return tau - cost(a, b); };
        auto result = convex_sweep::sweep_solve<true>(events, profit, model.radius(tau));
        assert(result.profit == tau - cost(pos[0], pos[1]));
        convex_sweep_dual::recover<I>(pos, std::vector<I>{1, 0, 0}, std::vector<I>{0, 1, 1},
            tau, model.radius(tau), cost, result.matches, W(result.profit));
    }
    {
        // Avoid forming q^p/M^p separately: both can overflow at large p.
        monge_cost::GridPower model;
        std::vector<double> pos;
        model.reset(std::vector<double>{0, .999, 1}, 1000000, 1000, pos);
        assert(model.cost(pos[0], pos[2]) > 990000);
        assert(model.cost(pos[0], pos[2]) / 1000000.0 <= 1 + model.error());
        assert(!(W(model.cost(pos[0], pos[1])) + W(model.cost(pos[1], pos[2]))
            < W(model.cost(pos[0], pos[0])) + W(model.cost(pos[1], pos[1]))));
    }
    std::mt19937 rng(20261013);
    for (size_t trial = 0; trial < 1000; ++trial) {
        const double p = std::vector<double>{1, 1.00000001, 1.1, 1.5, 2, 2.5, 3, 5, 20}[trial % 9];
        const I scale = 1 + rng() % 1000000;
        std::vector<double> real(6), pos;
        std::vector<I> e(6), t(6);
        for (size_t i = 0; i < real.size(); ++i) {
            real[i] = (i ? real[i - 1] : 0) + (rng() % 1000) / 10000.0;
            e[i] = rng() % 3; t[i] = rng() % 2;
        }
        monge_cost::GridPower model;
        model.reset(real, scale, p, pos);
        for (size_t i = 0; i < pos.size(); ++i) for (size_t k = i; k < pos.size(); ++k)
            for (size_t j = 0; j < pos.size(); ++j) for (size_t l = j; l < pos.size(); ++l)
                assert(!(W(model.cost(pos[i], pos[l])) + W(model.cost(pos[k], pos[j]))
                    < W(model.cost(pos[i], pos[j])) + W(model.cost(pos[k], pos[l]))));
        for (size_t i = 0; i < pos.size(); ++i) for (size_t j = 0; j < pos.size(); ++j)
            assert(std::abs(model.cost(pos[i], pos[j]) / double(scale)
                - std::pow(std::abs(real[i] - real[j]), p)) <= model.error());
        const I tau = rng() % 1000;
        check<convex_sweep::PLF>(pos, e, t, tau, model);
        check<convex_sweep::TreePLF>(pos, e, t, tau, model);
        check<convex_sweep::AugmentedPLF>(pos, e, t, tau, model);
        // Keep the same certificate model even when a previously zero target
        // receives mass; the quantizer is geometry-only.
        t[trial % t.size()] += 1;
        check<convex_sweep::PLF>(pos, e, t, tau, model);
    }
    // The two geometries which broke independently rounded pair costs.
    for (double p : {1, 2}) {
        std::vector<double> real = p == 1 ? std::vector<double>{0, .588, 1.166, 1.474}
            : std::vector<double>{0, .202, .387, .744, .827, 1.641};
        std::vector<I> e(real.size()), t(real.size());
        e[0] = e[1] = e[2] = 1;
        for (size_t j = p == 1 ? 1 : 2; j < t.size(); ++j) t[j] = 1;
        monge_cost::GridPower model;
        std::vector<double> pos;
        model.reset(real, 3, p, pos);
        check<convex_sweep::PLF>(pos, e, t, p == 1 ? 17 : 10, model);
    }
    std::puts("1000 Monge/oracle/error-bound cases, all stores and rounding regressions passed");
}
