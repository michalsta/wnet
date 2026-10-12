#include "sweep_options.hpp"
#include <cassert>
using namespace sweep_options;
I oracle(const Input &in, std::vector<I> t = {}) {
    if (t.empty())
        t = in.t;
    std::vector<size_t> rows, cols;
    for (size_t i = 0; i < in.pos.size(); ++i) {
        for (I k = 0; k < in.e[i]; ++k)
            rows.push_back(i);
        for (I k = 0; k < t[i]; ++k)
            cols.push_back(i);
    }
    std::vector<I> dp(size_t(1) << cols.size(), -100000000);
    dp[0] = 0;
    for (size_t i : rows) {
        auto next = dp;
        for (size_t mask = 0; mask < dp.size(); ++mask)
            for (size_t j = 0; j < cols.size(); ++j)
                if (!(mask & (size_t(1) << j)))
                    next[mask | (size_t(1) << j)] =
                        std::max(next[mask | (size_t(1) << j)],
                                 dp[mask] + in.tau - in.cost(in.pos[i], in.pos[cols[j]]));
        dp = std::move(next);
    }
    return *std::max_element(dp.begin(), dp.end());
}
int main() {
    {
        Input in;
        in.p = 2;
        in.tau = 50;
        in.scale = 1;
        for (size_t i = 0; i < 13; ++i) {
            in.pos.push_back(double(i));
            in.e.push_back(i == 0 ? 6 : 0);
            in.t.push_back(i ? 1 : 0);
        }
        auto r = cost_scaling(in, 8000000, true);
        validate(in, r, true);
        assert(r.exchanges > 1 && r.profit == W(oracle(in)));
    }
    std::mt19937 rng(4821);
    for (size_t trial = 0; trial < 500; ++trial) {
        Input in;
        size_t n = 4;
        in.p = 1 + rng() % 2;
        in.scale = 1 + rng() % 30;
        in.tau = 1 + rng() % 30;
        for (size_t i = 0; i < n; ++i) {
            in.pos.push_back((i ? in.pos.back() : 0) + double(rng() % 1000) / 1000);
            in.e.push_back(rng() % 3);
            in.t.push_back(rng() % 3);
        }
        I best = oracle(in);
        for (double h : {.01, .001}) {
            auto r = monge(in, h);
            auto model = in;
            model.scale = 1;
            model.tau = std::llround((long double)in.tau / in.scale / (in.p == 1 ? h : h * h));
            for (auto &x : model.pos)
                x = std::llround(x / h);
            assert(matching_profit(model, r.matches) == W(oracle(model)));
            auto native = monge(in, h, false);
            validate(model, native, true);
            assert(native.profit == W(oracle(model)));
            assert(native.original_profit == matching_profit(in, native.matches));
            auto sweep_native = monge(in, h, false, true);
            validate(model, sweep_native, true);
            assert(sweep_native.profit == native.profit);
        }
        std::vector<Result> results{
            baseline(in),    cached(in),     cost_scaling(in), cost_scaling(in, 8000000, true),
            approximate(in), monge(in, .01), monge(in, .001)};
        if (in.p == 1) {
            results.push_back(slope_chain(in));
            results.push_back(slope_chain(in, true, true));
            results.push_back(monge(in, .001, true, false, true));
        }
        for (size_t k = 0; k < results.size(); ++k) {
            auto &r = results[k];
            validate(in, r, k < 4);
            assert(!(W(best) < r.profit) && !(r.dual_profit < W(best)));
            if (k < 4) {
                auto cost = [&](double a, double b) { return in.cost(a, b); };
                auto gains = convex_sweep_dual::repaired_gains<I>(
                    in.pos, in.e, in.t, in.tau, in.radius(), cost, r.matches, r.alpha, r.beta);
                for (size_t j = 0; j < n; ++j) {
                    auto probe = in;
                    probe.t[j] += 1;
                    assert(gains[j] == oracle(probe) - best);
                }
            }
            for (size_t j = 0; j < n; ++j) {
                auto probe = in;
                probe.t[j] += 1;
                I opt = oracle(probe);
                assert(!(dual_profit(probe, r.alpha, r.beta) < W(opt)));
            }
        }
    }
    // The reduced non-Monge failures must be repaired exactly.
    for (int p : {1, 2}) {
        Input in;
        in.p = p;
        in.scale = 3;
        in.tau = p == 1 ? 17 : 10;
        in.pos = p == 1 ? std::vector<double>{.337, .925, 1.503, 1.811}
                        : std::vector<double>{.19, .392, .577, .934, 1.017, 1.831};
        in.e = p == 1 ? std::vector<I>{1, 1, 1, 0} : std::vector<I>{1, 1, 1, 0, 0, 0};
        in.t = p == 1 ? std::vector<I>{0, 1, 1, 1} : std::vector<I>{0, 0, 1, 1, 1, 1};
        for (auto r : {baseline(in), cached(in), cost_scaling(in)}) {
            validate(in, r, true);
            assert(r.profit == W(p == 1 ? 47 : 29));
        }
    }
    std::puts(
        "500 weighted oracle cases, all original-model dual probes and both regressions passed");
}
