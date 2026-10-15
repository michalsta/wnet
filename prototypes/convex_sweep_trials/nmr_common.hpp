#ifndef WNET_NMR_TRIAL_COMMON_HPP
#define WNET_NMR_TRIAL_COMMON_HPP
#include "wnet/convex_sweep_dual.hpp"
#include "wnet/convex_sweep_events.hpp"
#include "wnet/monge_cost.hpp"
#include <fstream>
#include <numeric>
namespace nmr_trials {
using I = int64_t;
using W = wide_int::wide_t;
struct Input {
    std::vector<double> pos;
    std::vector<I> e, t;
    std::vector<uint8_t> theo;
    monge_cost::GridPower model;
    I tau = 0;
    double p = 2;
    void grid(const std::vector<double> &real, I scale, double order, I reward) {
        p = order;
        tau = reward;
        model.reset(real, scale, p, pos);
    }
};
struct Coordinates {
    std::vector<double> pos;
    std::vector<I> e, t;
    std::vector<bool> theo;
    std::vector<size_t> original;
    explicit Coordinates(const Input &in) {
        for (size_t i = 0; i < in.pos.size(); ++i) {
            if (pos.empty() || pos.back() != in.pos[i]) {
                pos.push_back(in.pos[i]);
                e.push_back(0);
                t.push_back(0);
                theo.push_back(false);
            }
            e.back() += in.e[i];
            t.back() += in.t[i];
            theo.back() = theo.back() || in.theo[i] || in.t[i] > 0;
            original.push_back(pos.size() - 1);
        }
    }
};
template <bool Map> auto events(const Input &in, bool mirror = false, bool probes = false) {
    convex_sweep::GroupedEvents<Map> g;
    auto emit = [&](size_t i) {
        double z = mirror ? -in.pos[i] : in.pos[i];
        if (in.e[i])
            g.add(z, 0, in.e[i], i);
        if (in.t[i] || (probes && in.theo[i]))
            g.add(z, 1, in.t[i], i);
    };
    if (mirror)
        for (size_t i = in.pos.size(); i-- > 0;)
            emit(i);
    else
        for (size_t i = 0; i < in.pos.size(); ++i)
            emit(i);
    g.finish();
    return g;
}
// Same versioned native fixture format as the preceding benchmark series.
struct Fixture {
    I scale, price, base;
    double intensity, p;
    std::vector<Input> components;
    size_t original_events = 0, grouped_events = 0;
    template <class T> static T read(std::istream &f) {
        T x{};
        f.read(reinterpret_cast<char *>(&x), sizeof x);
        if (!f)
            throw std::runtime_error("truncated fixture");
        return x;
    }
    template <class T> static auto readv(std::istream &f, size_t n) {
        std::vector<T> x(n);
        f.read(reinterpret_cast<char *>(x.data()), sizeof(T) * n);
        if (!f)
            throw std::runtime_error("truncated fixture vector");
        return x;
    }
    explicit Fixture(const char *path) {
        std::ifstream f(path, std::ios::binary);
        if (read<uint64_t>(f) != 0x31545343)
            throw std::runtime_error("fixture version");
        auto n = read<uint64_t>(f);
        scale = read<I>(f);
        intensity = read<double>(f);
        price = read<I>(f);
        p = read<double>(f);
        base = read<I>(f);
        for (size_t j = 0; j < n; ++j) {
            auto m = read<uint64_t>(f);
            auto real = readv<double>(f, m);
            Input in;
            in.e = readv<I>(f, m);
            in.t = readv<I>(f, m);
            in.theo = readv<uint8_t>(f, m);
            in.grid(real, scale, p, 2 * price);
            for (size_t i = 0; i < m; ++i)
                original_events += (in.e[i] > 0) + (in.t[i] > 0);
            grouped_events += events<false>(in).events.size();
            components.push_back(std::move(in));
        }
    }
};
struct Certificate {
    std::vector<I> alpha, beta;
};
inline Certificate recover(const Input &in, const convex_sweep::GroupedEvents<true> &ev,
                           const std::vector<convex_sweep::Match> &matching, I profit,
                           bool grouped) {
    if (!grouped) {
        auto matches = ev.expand(matching);
        auto [a, b] = convex_sweep_dual::recover<I>(
            in.pos, in.e, in.t, in.tau, in.model.radius(in.tau),
            [&](double a, double b) { return in.model.cost(a, b); }, matches, W(profit));
        return {std::move(a), std::move(b)};
    }
    Coordinates c(in);
    std::vector<size_t> index(ev.events.size());
    for (size_t i = 0; i < ev.events.size(); ++i)
        index[i] =
            size_t(std::lower_bound(c.pos.begin(), c.pos.end(), ev.events[i].pos) - c.pos.begin());
    std::vector<convex_sweep::Match> matches;
    matches.reserve(matching.size());
    for (auto m : matching)
        matches.push_back({index[m.empirical], index[m.theoretical], m.count});
    auto [a, b] = convex_sweep_dual::recover<I>(
        c.pos, c.e, c.t, in.tau, in.model.radius(in.tau),
        [&](double a, double b) { return in.model.cost(a, b); }, matches, W(profit));
    Certificate out;
    out.alpha.reserve(in.pos.size());
    out.beta.reserve(in.pos.size());
    for (auto j : c.original) {
        out.alpha.push_back(a[j]);
        out.beta.push_back(b[j]);
    }
    return out;
}
inline void require(bool ok, const char *what) {
    if (!ok)
        throw std::runtime_error(what);
}
} // namespace nmr_trials
#endif
