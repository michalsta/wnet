// Independent capacitated min-cost-flow oracle, grouping metamorphisms and
// exact +1 sensitivities. Assert-enabled standalone; no Python/LEMON dependency.
#include "wnet/convex_sweep_dual.hpp"
#include "wnet/convex_sweep_events.hpp"
#include "wnet/monge_cost.hpp"
#include <iostream>
#include <numeric>
#include <random>
using namespace convex_sweep;
struct Input {
    std::vector<double> x;
    std::vector<i64> e, t;
    std::vector<bool> theo;
    monge_cost::GridPower model;
    i64 tau;
};
void require(bool ok, const char *what) {
    if (!ok)
        throw std::runtime_error(what);
}
// Every pair is present. No monotone-matching restriction or sweep recurrence.
i64 oracle(const Input &in) {
    struct Edge {
        size_t to, rev;
        i64 cap, cost;
    };
    const size_t n = in.x.size(), source = 2 * n, sink = source + 1, N = sink + 1;
    std::vector<std::vector<Edge>> g(N);
    auto add = [&](size_t a, size_t b, i64 cap, i64 cost) {
        g[a].push_back({b, g[b].size(), cap, cost});
        g[b].push_back({a, g[a].size() - 1, 0, -cost});
    };
    for (size_t i = 0; i < n; ++i) {
        add(source, i, in.e[i], 0);
        add(n + i, sink, in.t[i], 0);
        for (size_t j = 0; j < n; ++j)
            add(i, n + j, std::min(in.e[i], in.t[j]), in.model.cost(in.x[i], in.x[j]) - in.tau);
    }
    i64 total = 0;
    for (;;) {
        std::vector<i64> d(N, std::numeric_limits<i64>::max() / 4);
        d[source] = 0;
        std::vector<size_t> node(N, N), arc(N);
        for (size_t it = 0; it < N; ++it) {
            bool changed = false;
            for (size_t u = 0; u < N; ++u)
                if (d[u] < std::numeric_limits<i64>::max() / 4)
                    for (size_t j = 0; j < g[u].size(); ++j) {
                        auto a = g[u][j];
                        if (a.cap && d[a.to] > d[u] + a.cost) {
                            d[a.to] = d[u] + a.cost;
                            node[a.to] = u;
                            arc[a.to] = j;
                            changed = true;
                        }
                    }
            if (!changed)
                break;
            require(it + 1 < N, "negative residual cycle in independent oracle");
        }
        if (d[sink] >= 0)
            break;
        i64 amount = std::numeric_limits<i64>::max();
        for (size_t v = sink; v != source; v = node[v])
            amount = std::min(amount, g[node[v]][arc[v]].cap);
        for (size_t v = sink; v != source; v = node[v]) {
            auto &a = g[node[v]][arc[v]];
            a.cap -= amount;
            g[v][a.rev].cap += amount;
        }
        total -= amount * d[sink];
    }
    return total;
}
template <bool Mapping> auto events(const Input &in, bool mirror = false, bool probes = true) {
    GroupedEvents<Mapping> g;
    auto emit = [&](size_t i) {
        double x = mirror ? -in.x[i] : in.x[i];
        if (in.e[i])
            g.add(x, 0, in.e[i], i);
        if (in.t[i] || (probes && in.theo[i]))
            g.add(x, 1, in.t[i], i);
    };
    if (mirror)
        for (size_t i = in.x.size(); i-- > 0;)
            emit(i);
    else
        for (size_t i = 0; i < in.x.size(); ++i)
            emit(i);
    g.finish();
    return g;
}
template <class F> i64 solve(const Input &in) {
    auto g = events<false>(in, false, false);
    return sweep_solve<false, F>(
        g.events, [&](double a, double b) { return in.tau - in.model.cost(a, b); },
        in.model.radius(in.tau));
}
template <class F> std::vector<i64> gains(const Input &in) {
    std::vector<i64> result(in.x.size());
    for (bool mirror : {false, true}) {
        auto g = events<true>(in, mirror);
        auto profit = [&](double a, double b) { return in.tau - in.model.cost(a, b); };
        std::vector<BasicSnapshot<F>> snaps;
        std::vector<i64> composed;
        auto base = forward_store(g.events, profit, in.model.radius(in.tau), snaps);
        adjoint_compose(g.events, profit, snaps, composed);
        for (size_t i = 0; i < g.events.size(); ++i)
            if (g.events[i].side == 1)
                for (auto part : g.parts[i])
                    result[part.index] = std::max(result[part.index], composed[i] - base);
    }
    return result;
}
template <class F> void check(const Input &in, bool marginal = true) {
    auto expected = oracle(in);
    require(solve<F>(in) == expected, "grouped objective");
    // Pruning and sweep orientation must not affect the optimum.
    for (bool mirror : {false, true}) {
        auto unpruned = events<false>(in, mirror, false);
        auto pi = [&](double a, double b) { return in.tau - in.model.cost(a, b); };
        require(sweep_solve<false, F>(unpruned.events, pi, std::numeric_limits<double>::max()) ==
                    expected,
                "unpruned orientation");
    }
    auto g = events<true>(in, false, false);
    auto profit = [&](double a, double b) { return in.tau - in.model.cost(a, b); };
    auto trace = sweep_solve<true, F>(g.events, profit, in.model.radius(in.tau));
    require(trace.profit == expected, "traced objective");
    auto matching = g.expand(trace.matches);
    std::vector<i64> e(in.x.size()), t(in.x.size());
    i64 value = 0;
    for (auto m : matching) {
        require(m.count > 0, "zero match");
        e.at(m.empirical) += m.count;
        t.at(m.theoretical) += m.count;
        value += m.count * profit(in.x[m.empirical], in.x[m.theoretical]);
    }
    require(value == expected, "expanded matching profit");
    for (size_t i = 0; i < e.size(); ++i)
        require(e[i] <= in.e[i] && t[i] <= in.t[i], "expanded matching capacity");
    auto cost = [&](double a, double b) { return in.model.cost(a, b); };
    auto [alpha, beta] =
        convex_sweep_dual::recover<i64>(in.x, in.e, in.t, in.tau, in.model.radius(in.tau), cost,
                                        matching, wide_int::wide_t(expected));
    wide_int::wide_t dual = 0;
    for (size_t i = 0; i < e.size(); ++i) {
        require(alpha[i] <= 0 && beta[i] <= 0, "dual sign");
        dual += wide_int::wide_t(in.e[i]) * alpha[i] + wide_int::wide_t(in.t[i]) * beta[i];
        for (size_t j = 0; j < e.size(); ++j)
            require(alpha[i] + beta[j] <= -profit(in.x[i], in.x[j]), "dual pair infeasible");
    }
    require(dual == -wide_int::wide_t(expected), "duality gap");
    if (marginal) {
        auto gain = gains<F>(in);
        for (size_t i = 0; i < e.size(); ++i)
            if (in.theo[i]) {
                auto more = in;
                ++more.t[i];
                auto want = oracle(more) - expected;
                if (gain[i] != want) {
                    std::cerr << "probe=" << i << " tau=" << in.tau << " got=" << gain[i]
                              << " expected=" << want << "\n";
                    for (size_t j = 0; j < e.size(); ++j)
                        std::cerr << in.x[j] << " " << in.e[j] << " " << in.t[j] << " "
                                  << in.theo[j] << "\n";
                    throw std::runtime_error("one-unit marginal");
                }
            }
    }
}
Input make(std::vector<double> x, std::vector<i64> e, std::vector<i64> t, i64 tau = 41,
           double p = 2, i64 scale = 1) {
    Input in;
    in.e = e;
    in.t = t;
    in.theo.assign(x.size(), true);
    in.tau = tau;
    in.model.reset(x, scale, p, in.x);
    return in;
}
int main(int argc, char **argv) {
    size_t trials = argc > 1 ? std::stoul(argv[1]) : 3000;
    std::mt19937_64 rng(822);
    // Target blocks bracket empirical blocks at a tie in both orientations.
    // The old grouped order misses a gain of four at coordinate 14.
    auto regression = make({2, 3, 5, 6, 6, 8, 8, 9, 11, 12, 14, 14, 14, 14},
                           {0, 0, 0, 10, 2, 0, 0, 11, 0, 0, 0, 17, 9, 0},
                           {13, 5, 3, 0, 0, 12, 11, 0, 8, 13, 15, 0, 0, 4});
    auto minimal = make({0, 0, 0}, {0, 5, 0}, {3, 0, 1});
    check<PLF>(minimal);
    check<TreePLF>(minimal);
    check<AugmentedPLF>(minimal);
    check<PLF>(make({}, {}, {}, 0));
    check<PLF>(make({0, 1, 2}, {1000000000, 0, 0}, {0, 0, 1000000000}, 1000000));
    check<PLF>(regression);
    check<TreePLF>(regression);
    check<AugmentedPLF>(regression);
    for (size_t it = 0; it < trials; ++it) {
        size_t n = 1 + rng() % 18;
        std::vector<double> x(n);
        std::vector<i64> e(n), t(n);
        i64 scale = it % 3 == 0 ? 1 : 1 + rng() % 1000000;
        double p = std::vector<double>{1, 1.125, 1.5, 2, 3}[it % 5];
        for (size_t i = 0; i < n; ++i) {
            x[i] = (i ? x[i - 1] : 5) + (rng() % 4) * (it % 3 == 0 ? 1 : .03);
            e[i] = rng() % 31;
            t[i] = rng() % 31;
            if (it % 2) {
                if (rng() % 2)
                    e[i] = 0;
                else
                    t[i] = 0;
            }
            if (it % 7 == 0) {
                e[i] *= 1000000;
                t[i] *= 1000000;
            }
        }
        auto in = make(x, e, t, 1 + rng() % (it % 3 == 0 ? 50 : 1000000), p, scale);
        for (size_t i = 0; i < n; ++i)
            in.theo[i] = in.t[i] > 0 || rng() % 2;
        try {
            check<PLF>(in);
            if (it < 300) {
                check<TreePLF>(in);
                check<AugmentedPLF>(in);
            }
            // Split every peak into fragments, shuffle opposite-side roles
            // inside ties, and activate zero-count targets at new locations.
            if (it < 500) {
                Input split = in;
                split.x.clear();
                split.e.clear();
                split.t.clear();
                split.theo.clear();
                for (size_t i = 0; i < n; ++i) {
                    const i64 a = rng() % (in.e[i] + 1), b = rng() % (in.t[i] + 1);
                    split.x.insert(split.x.end(), 3, in.x[i]);
                    split.e.insert(split.e.end(), {0, a, in.e[i] - a});
                    split.t.insert(split.t.end(), {b, 0, in.t[i] - b});
                    split.theo.insert(split.theo.end(), 3, in.theo[i]);
                }
                check<PLF>(split);
            }
        } catch (const std::exception &ex) {
            std::cerr << "case=" << it << " p=" << p << " scale=" << scale << " " << ex.what()
                      << "\n";
            return 1;
        }
    }
    // Exhaust small capacities, including empty sides and exact radius ties.
    for (int code = 0; code < 729; ++code)
        for (i64 tau : {0, 1, 4, 5}) {
            int c = code;
            std::vector<i64> e(3), t(3);
            for (int j = 0; j < 3; ++j) {
                e[j] = c % 3;
                c /= 3;
                t[j] = c % 3;
                c /= 3;
            }
            auto in = make({0, 1, 2}, e, t, tau);
            check<PLF>(in);
        }
    std::cout << trials
              << " randomized cases, split/merged metamorphisms, all-storage regression, and 2916 "
                 "exhaustive capacity cases passed\n";
    return 0;
}
