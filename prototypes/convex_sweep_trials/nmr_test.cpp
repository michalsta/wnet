#include "nmr_trials.hpp"
#include <iostream>
#include <random>
using namespace nmr_trials;
// Independent full bipartite capacitated SSP oracle. It does not use LEMON,
// monotone matching, the sweep recurrence, or the residual sensitivity code.
I oracle(const Input &in) {
    struct Edge {
        size_t to, rev;
        I cap, cost;
    };
    size_t n = in.pos.size(), src = 2 * n, sink = src + 1, N = sink + 1;
    std::vector<std::vector<Edge>> g(N);
    auto add = [&](size_t a, size_t b, I cap, I cost) {
        g[a].push_back({b, g[b].size(), cap, cost});
        g[b].push_back({a, g[a].size() - 1, 0, -cost});
    };
    for (size_t i = 0; i < n; ++i) {
        add(src, i, in.e[i], 0);
        add(n + i, sink, in.t[i], 0);
        for (size_t j = 0; j < n; ++j) {
            const W cost = in.model.wide_cost(in.pos[i], in.pos[j]);
            if (cost < W(in.tau))
                add(i, n + j, std::min(in.e[i], in.t[j]), static_cast<I>(cost) - in.tau);
        }
    }
    I total = 0;
    for (;;) {
        std::vector<I> d(N, INT64_MAX / 4);
        std::vector<size_t> node(N), arc(N);
        d[src] = 0;
        for (size_t k = 0; k < N; ++k) {
            bool changed = false;
            for (size_t u = 0; u < N; ++u)
                if (d[u] < INT64_MAX / 4)
                    for (size_t a = 0; a < g[u].size(); ++a) {
                        auto e = g[u][a];
                        if (e.cap && d[e.to] > d[u] + e.cost) {
                            d[e.to] = d[u] + e.cost;
                            node[e.to] = u;
                            arc[e.to] = a;
                            changed = true;
                        }
                    }
            if (!changed)
                break;
            require(k + 1 < N, "oracle negative cycle");
        }
        if (d[sink] >= 0)
            break;
        I count = INT64_MAX;
        for (size_t v = sink; v != src; v = node[v])
            count = std::min(count, g[node[v]][arc[v]].cap);
        for (size_t v = sink; v != src; v = node[v]) {
            auto &e = g[node[v]][arc[v]];
            e.cap -= count;
            g[v][e.rev].cap += count;
        }
        total -= count * d[sink];
    }
    return total;
}
void certificate(const Input &in, const Result &r) {
    require(r.certificate.alpha.size() == in.pos.size(), "certificate size");
    W sum = 0;
    for (size_t i = 0; i < in.pos.size(); ++i) {
        auto a = r.certificate.alpha[i], b = r.certificate.beta[i];
        require(a <= 0 && b <= 0, "dual signs");
        sum += W(a) * in.e[i] + W(b) * in.t[i];
        for (size_t j = 0; j < in.pos.size(); ++j)
            require(!(in.model.wide_cost(in.pos[i], in.pos[j]) - W(in.tau) <
                      W(a) + W(r.certificate.beta[j])),
                    "dual feasibility");
    }
    require(sum == -W(r.profit), "dual objective");
}
Input make(std::vector<double> x, std::vector<I> e, std::vector<I> t, I tau = 41, double p = 2,
           I scale = 1) {
    Input in;
    in.grid(x, scale, p, tau);
    in.e = e;
    in.t = t;
    in.theo.assign(x.size(), 1);
    return in;
}
void check(const Input &in, size_t seed) {
    I expected = oracle(in);
    std::vector<I> gain(in.pos.size());
    for (size_t i = 0; i < in.pos.size(); ++i)
        if (in.theo[i] || in.t[i]) {
            auto more = in;
            ++more.t[i];
            gain[i] = oracle(more) - expected;
        }
    for (auto name : {"baseline", "flow", "virtual", "quadratic", "groupdual", "checkpoint64",
                      "checkpoint256-groupdual", "virtual-checkpoint256", "quadratic-checkpoint256",
                      "virtual-cached", "quadratic-cached", "virtual-cached-checkpoint256",
                      "quadratic-cached-checkpoint256"}) {
        auto v = variant(name);
        if (v.quadratic && in.p != 2)
            continue;
        // Force replay boundaries inside small cases, not just at the endpoints.
        if (v.stride)
            v.stride = 1 + seed % 7;
        try {
            require(run(in, v, "solve").profit == expected, "ordinary profit");
            auto r = run(in, v, "dual");
            require(r.profit == expected, "traced profit");
            certificate(in, r);
            if (v.costflow) {
                Coordinates c(in);
                size_t pairs = 0;
                for (size_t i = 0; i < c.pos.size(); ++i)
                    if (c.e[i])
                        for (size_t j = 0; j < c.pos.size(); ++j)
                            if (c.t[j] || c.theo[j])
                                pairs += in.model.wide_cost(c.pos[i], c.pos[j]) < W(in.tau);
                require(r.pairs == pairs, "profitable pair census");
            }
            if (seed < 50 && v.stride) {
                auto cmp = run(in, v, "dual-compare");
                require(cmp.profit == expected, "recovery comparison profit");
                certificate(in, cmp);
            }
            auto g = run(in, v, "gains").gains;
            if (g != gain) {
                for (size_t i = 0; i < g.size(); ++i)
                    if (g[i] != gain[i])
                        std::cerr << "i=" << i << " got=" << g[i] << " expected=" << gain[i]
                                  << "\n";
                throw std::runtime_error("marginal mismatch");
            }
            if (seed < 50) {
                auto both = run(in, v, "both");
                require(both.profit == expected && both.gains == gain, "combined cost and gains");
            }
        } catch (const std::exception &e) {
            std::cerr << "variant=" << name << " stride=" << v.stride << " seed=" << seed
                      << " tau=" << in.tau << " p=" << in.p << " " << e.what() << "\n";
            for (size_t i = 0; i < in.pos.size(); ++i)
                std::cerr << in.pos[i] << " " << in.e[i] << " " << in.t[i] << " " << int(in.theo[i])
                          << "\n";
            throw;
        }
    }
}
void view_tests() {
    std::mt19937_64 rng(45);
    for (size_t test = 0; test < 10000; ++test) {
        const I K = 1 + rng() % 50, q = rng() % (K + 5);
        k::PLF f, t;
        for (auto *curve : {&f, &t}) {
            I rank = 0, value = I(rng() % 100) - 50;
            curve->vs.push_back({0, value});
            while (rank < K) {
                I span = std::min<I>(1 + rng() % 7, K - rank);
                rank += span;
                value += span * (I(rng() % 21) - 10);
                curve->vs.push_back({rank, value});
            }
            curve->normalize();
        }
        nmr_views::Sum sum(f, t);
        std::vector<I> keys;
        for (auto x : f.vs)
            keys.push_back(x.k);
        for (auto x : t.vs)
            keys.push_back(x.k);
        std::sort(keys.begin(), keys.end());
        keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
        size_t j = 0;
        for (auto x : sum.vs) {
            require(j < keys.size() && x.k == keys[j++] && x.v == f.eval(x.k) + t.eval(x.k),
                    "virtual union iterator");
        }
        require(j == keys.size(), "virtual union size");
        auto out = k::window_sub_view<k::PLF>(sum, q, t);
        nmr_views::Sum<k::PLF, std::remove_cvref_t<decltype(t)>, true> cached(f, t);
        auto cached_out = k::window_sub_view<k::PLF>(cached, q, t);
        for (I x = 0; x <= K; ++x) {
            I best = k::NEG_INF;
            for (I y = x; y <= std::min(K, x + q); ++y)
                best = std::max(best, f.eval(y) + t.eval(y));
            require(out.eval(x) == best - t.eval(x), "virtual window brute force");
            require(cached_out.eval(x) == best - t.eval(x), "cached virtual brute force");
        }
    }
    for (size_t test = 0; test < 1000; ++test) {
        std::vector<k::Event> ev;
        I mass = 0;
        for (size_t j = 0; j < 6; ++j) {
            I count = 1 + rng() % 8;
            ev.push_back({double(j), 0, count, j});
            mass += count;
        }
        sweep_trials::MomentBank bank(ev);
        sweep_trials::FastMomentCurve t(bank.sides[0], mass, mass, 7, 1000);
        k::PLF f;
        f.vs.push_back({0, 0});
        I value = 0;
        for (I rank = 1; rank <= mass; ++rank) {
            value += I(rng() % 21) - 10;
            f.vs.push_back({rank, value});
        }
        f.normalize();
        nmr_views::Sum sum(f, t);
        I q = rng() % (mass + 5);
        auto out = k::window_sub_view<k::PLF>(sum, q, t);
        nmr_views::Sum<k::PLF, std::remove_cvref_t<decltype(t)>, true> cached(f, t);
        auto cached_out = k::window_sub_view<k::PLF>(cached, q, t);
        for (I x = 0; x <= mass; ++x) {
            I best = k::NEG_INF;
            for (I y = x; y <= std::min(mass, x + q); ++y)
                best = std::max(best, f.eval(y) + t.eval(y));
            require(out.eval(x) == best - t.eval(x), "quadratic view brute force");
            require(cached_out.eval(x) == best - t.eval(x), "cached quadratic brute force");
        }
    }
}
int main(int argc, char **argv) {
    view_tests();
    size_t trials = argc > 1 ? std::stoul(argv[1]) : 3000;
    std::mt19937_64 rng(1241);
    check(make({}, {}, {}, 0), 0);
    check(make({0, 0, 0}, {0, 5, 0}, {3, 0, 1}), 1);
    check(make({0, 0, 1}, {5000000000000001LL, 0, 0}, {0, 5000000000000000LL, 0}, 2), 2);
    // 256-bit moment cancellation with tiny distances at a large origin/span.
    check(make({0, 999999999999., 1000000000000.}, {0, 1000000000000000LL, 0},
               {0, 0, 999999999999999LL}, 2),
          3);
    for (size_t it = 0; it < trials; ++it) {
        size_t n = 1 + rng() % 16;
        std::vector<double> x(n);
        std::vector<I> e(n), t(n);
        for (size_t i = 0; i < n; ++i) {
            x[i] = (i ? x[i - 1] : 0) + double(rng() % 4) * (it % 3 ? 0.04 : 1);
            e[i] = rng() % 20;
            t[i] = rng() % 20;
            if (it % 2) {
                if (rng() % 2)
                    e[i] = 0;
                else
                    t[i] = 0;
            }
            if (it % 9 == 0) {
                e[i] *= 1000000;
                t[i] *= 1000000;
            }
        }
        double p = std::vector<double>{1, 1.125, 1.5, 2, 3}[it % 5];
        auto in = make(x, e, t, it % 17 ? 1 + rng() % 100 : 0, p, it % 3 ? 1 + rng() % 1000 : 1);
        for (size_t i = 0; i < n; ++i)
            in.theo[i] = in.t[i] > 0 || rng() % 2;
        check(in, it + 10);
        if (it < 100) {
            auto split = in;
            split.pos.clear();
            split.e.clear();
            split.t.clear();
            split.theo.clear();
            for (size_t i = 0; i < n; ++i) {
                I a = rng() % (in.e[i] + 1), b = rng() % (in.t[i] + 1);
                split.pos.insert(split.pos.end(), 3, in.pos[i]);
                split.e.insert(split.e.end(), {0, a, in.e[i] - a});
                split.t.insert(split.t.end(), {b, 0, in.t[i] - b});
                split.theo.insert(split.theo.end(), 3, in.theo[i]);
            }
            check(split, it + 10);
        }
        if (it % 500 == 0)
            std::cerr << "cases=" << it << "\n";
    }
    // Long exact instance crosses even the 1024-event replay boundaries, with
    // pruning immediately at boundaries, and includes a zero-count end probe.
    auto longin = make({}, {}, {}, 2);
    longin.pos.resize(1101);
    longin.e.assign(1101, 1);
    longin.t.assign(1101, 1);
    longin.theo.assign(1101, 1);
    std::vector<double> x(1101);
    for (size_t i = 0; i < x.size(); ++i)
        x[i] = 3 * i;
    longin.grid(x, 1, 2, 2);
    longin.t.back() = 0;
    auto baseline = run(longin, variant("baseline"), "both");
    require(baseline.profit == 2200 && baseline.gains.back() == 2, "long reference");
    for (size_t stride :
         {size_t(1), size_t(2), size_t(7), size_t(64), size_t(256), size_t(1024), size_t(10000)})
        for (auto name : {"checkpoint256-groupdual", "virtual-checkpoint256",
                          "quadratic-checkpoint256", "virtual-cached", "quadratic-cached",
                          "virtual-cached-checkpoint256", "quadratic-cached-checkpoint256"}) {
            auto v = variant(name);
            v.stride = stride;
            auto out = run(longin, v, "both");
            require(out.profit == baseline.profit && out.gains == baseline.gains,
                    "long checkpoint parity");
            certificate(longin, run(longin, v, "dual"));
        }
    bool rejected = false;
    try {
        flow(make({0}, {10}, {10}, 2), true, true, 0);
    } catch (const std::length_error &) {
        rejected = true;
    }
    require(rejected, "pair budget must reject rather than truncate");
    std::cout << trials
              << " independent flow-oracle cases, 10000 virtual + 1000 quadratic view tests, all "
                 "variants/certificates/+1 gains, split "
                 "events, wide moments and replay boundaries passed\n";
}
