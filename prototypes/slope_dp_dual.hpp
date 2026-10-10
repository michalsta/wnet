#ifndef WNET_PROTOTYPE_SLOPE_DP_DUAL_HPP
#define WNET_PROTOTYPE_SLOPE_DP_DUAL_HPP

// Standalone benchmarking wrapper around production ConvexPL and dual recovery.
#include "wnet/decompositable_graph.hpp"
#include <numeric>
#include <span>
#include <stdexcept>

namespace dp_dual_prototype {
using I = int64_t;
using W = slopedp_detail::wide_t;

struct Input {
    std::vector<I> e, t, gap;
    // Static cluster metadata, corresponding to the production DP cache.
    std::vector<size_t> start;
    std::vector<I> cluster_gap;
    void prepare() {
        if (e.size() != t.size() || gap.size() != (e.empty() ? 0 : e.size() - 1))
            throw std::invalid_argument("inconsistent chain sizes");
        start.clear(); cluster_gap.clear();
        start.push_back(0);
        for (size_t i = 1; i < e.size(); ++i) {
            if (gap[i - 1] < 0) throw std::invalid_argument("negative gap");
            if (gap[i - 1] > 0) {
                start.push_back(i); cluster_gap.push_back(gap[i - 1]);
            }
        }
        if (!e.empty()) start.push_back(e.size());
        for (size_t i = 0; i < e.size(); ++i)
            if (e[i] < 0 || t[i] < 0) throw std::invalid_argument("negative capacity");
    }
};

struct Primal {
    std::vector<I> u, y, s;  // matched empirical/theoretical mass and net gap flows
    I tau = 0;
    W profit_cost = 0;      // transport - tau * matched_mass
};
struct ClusterPrimal {
    std::vector<I> e, t, u, y, s;
};
template<bool Retain> struct Result : Primal {};
template<> struct Result<true> : Primal { ClusterPrimal clusters; };

// Mirrors the production DP's cluster forward pass, backward flow recovery,
// and expansion. The only Retain=true operation is moving already-computed
// cluster vectors into the result. Retain=false has no recording branch or
// extra allocation in the per-cluster/per-node loops.
template<bool Retain>
Result<Retain> solve(const Input& in, I tau) {
    if (tau < 0) throw std::invalid_argument("negative matching reward");
    const size_t P = in.e.size(), C = in.start.size() - 1;
    Result<Retain> out;
    out.tau = tau;
    out.u.assign(P, 0); out.y.assign(P, 0); out.s.assign(P ? P - 1 : 0, 0);
    std::vector<I> ce(C, 0), ct(C, 0), lo(C), hi(C), vv(C);
    for (size_t c = 0; c < C; ++c)
        for (size_t p = in.start[c]; p < in.start[c + 1]; ++p) {
            ce[c] += in.e[p]; ct[c] += in.t[p];
        }
    if (tau == 0) {
        // Production skips the matching DP when there is no profit to earn.
        if constexpr (Retain) {
            vv.resize(C ? C - 1 : 0);
            out.clusters = {std::move(ce), std::move(ct), std::move(lo),
                            std::move(hi), std::move(vv)};
        }
        return out;
    }
    slopedp_detail::ConvexPL<I> dp;
    const I wall = std::numeric_limits<I>::max() / 1024;
    dp.insL(0, wall); dp.insR(0, wall);
    for (size_t c = 0; c < C; ++c) {
        vv[c] = ce[c] - ct[c];
        auto [l, h] = dp.convV(tau, vv[c], ct[c]);
        lo[c] = l; hi[c] = h;
        dp.minval += -W(tau) * ct[c];
        if (c + 1 < C) dp.addAbs(0, in.cluster_gap[c]);
    }
    out.profit_cost = dp.evalAt(0);
    std::vector<I> uc(C, 0), yc(C, 0), sc(C ? C - 1 : 0, 0);
    I cur = 0;
    for (size_t cr = C; cr-- > 0;) {
        I prev = std::clamp(cur - vv[cr], lo[cr], hi[cr]);
        if (cur - prev < -ct[cr]) prev = cur + ct[cr];
        if (cur - prev > ce[cr]) prev = cur - ce[cr];
        if (cr == 0) prev = 0;
        I a = cur - prev;
        yc[cr] = std::max<I>(std::min(ct[cr], ce[cr] - a), 0);
        uc[cr] = std::clamp(a + yc[cr], I(0), ce[cr]);
        if (cr > 0) sc[cr - 1] = prev;
        cur = prev;
    }
    for (size_t c = 0; c < C; ++c) {
        I ul = uc[c], yl = yc[c], run = c ? sc[c - 1] : 0;
        for (size_t p = in.start[c]; p < in.start[c + 1]; ++p) {
            out.u[p] = std::min(in.e[p], ul); ul -= out.u[p];
            out.y[p] = std::min(in.t[p], yl); yl -= out.y[p];
            if (p + 1 < in.start[c + 1]) {
                run += out.u[p] - out.y[p]; out.s[p] = run;
            }
        }
        if (c + 1 < C) out.s[in.start[c + 1] - 1] = sc[c];
    }
    if constexpr (Retain) {
        out.clusters = {std::move(ce), std::move(ct), std::move(uc),
                        std::move(yc), std::move(sc)};
    }
    return out;
}

// Matching LP: min sum(gap_i |s_i|) - tau sum(y_i), subject to
// u_i-y_i+s_{i-1}-s_i=0, 0<=u<=e, 0<=y<=t, s_-1=s_last=0.
// Dual potential pi is restricted to [-tau,0]. Complementary slackness gives
// pi_i=0 if empirical capacity is slack and pi_i=-tau if theoretical capacity
// is slack. A positive net flow forces pi_{i+1}-pi_i=-gap_i; negative flow
// forces +gap_i; zero flow allows the interval [-gap_i,+gap_i].
// These are chain difference constraints, solved exactly in two linear passes.
// alpha_i=pi_i, beta_i=-tau-pi_i then form ONE common feasible matching dual.
inline std::vector<I> recover(std::span<const I> e, std::span<const I> t,
                              std::span<const I> u, std::span<const I> y,
                              std::span<const I> s, std::span<const I> gap, I tau) {
    return slopedp_dual::recover<I>(e, t, u, y, s, gap, tau);
}

inline std::vector<I> lazy_dual(const Input& in, const Primal& out) {
    return recover(in.e, in.t, out.u, out.y, out.s, in.gap, out.tau);
}
inline std::vector<I> cluster_dual(const Input& in, const Result<true>& out) {
    const auto& c = out.clusters;
    auto cp = recover(c.e, c.t, c.u, c.y, c.s, in.cluster_gap, out.tau);
    std::vector<I> pi(in.e.size());
    for (size_t c = 0; c < cp.size(); ++c)
        std::fill(pi.begin() + in.start[c], pi.begin() + in.start[c + 1], cp[c]);
    return pi;
}

// Always validate capacities, conservation, dual feasibility and exact
// primal-dual equality before describing the prototype's cut as certified.
inline W validate(const Input& in, const Primal& out, std::span<const I> pi) {
    if (pi.size() != in.e.size()) throw std::runtime_error("bad potential count");
    W primal = 0, dual = 0;
    for (size_t i = 0; i < pi.size(); ++i) {
        I left = i ? out.s[i - 1] : 0;
        I right = i + 1 < pi.size() ? out.s[i] : 0;
        if (out.u[i] - out.y[i] + left - right != 0 ||
            out.u[i] < 0 || out.u[i] > in.e[i] || out.y[i] < 0 || out.y[i] > in.t[i])
            throw std::runtime_error("invalid primal");
        if (pi[i] > 0 || pi[i] < -out.tau) throw std::runtime_error("dual outside box");
        if ((out.u[i] < in.e[i] && pi[i] != 0) ||
            (out.y[i] < in.t[i] && pi[i] != -out.tau))
            throw std::runtime_error("slack anchor complementarity");
        primal += -W(out.tau) * out.y[i];
        dual += W(pi[i]) * in.e[i] + W(-out.tau - pi[i]) * in.t[i];
        if (i + 1 < pi.size()) {
            I delta = pi[i + 1] - pi[i], g = in.gap[i];
            if (delta < -g || delta > g ||
                (out.s[i] > 0 && delta != -g) || (out.s[i] < 0 && delta != g))
                throw std::runtime_error("gap dual feasibility/complementarity");
            primal += W(out.s[i] >= 0 ? out.s[i] : -out.s[i]) * g;
        }
    }
    if (primal != dual || primal != out.profit_cost)
        throw std::runtime_error("primal-dual objective mismatch");
    return dual;
}
} // namespace dp_dual_prototype
#endif
