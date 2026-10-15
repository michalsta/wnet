#ifndef WNET_NMR_TRIALS_HPP
#define WNET_NMR_TRIALS_HPP
#include "nmr_flow.hpp"
#include "nmr_replay.hpp"
#include <chrono>
namespace nmr_trials {
struct Variant {
    std::string name;
    size_t stride = 0;
    bool grouped_dual = false, virt = false, quadratic = false, costflow = false;
};
inline Variant variant(std::string name) {
    Variant v{name};
    if (name == "baseline")
        return v;
    if (name == "virtual")
        v.virt = true;
    else if (name == "quadratic")
        v.virt = v.quadratic = true;
    else if (name == "groupdual")
        v.grouped_dual = true;
    else if (name == "flow")
        v.costflow = true;
    else if (name == "checkpoint64")
        v.stride = 64;
    else if (name == "checkpoint256")
        v.stride = 256;
    else if (name == "checkpoint1024")
        v.stride = 1024;
    else if (name == "checkpoint256-groupdual") {
        v.stride = 256;
        v.grouped_dual = true;
    } else if (name == "virtual-checkpoint256") {
        v.stride = 256;
        v.virt = true;
        v.grouped_dual = true;
    } else if (name == "quadratic-checkpoint256") {
        v.stride = 256;
        v.virt = v.quadratic = true;
        v.grouped_dual = true;
    } else
        throw std::invalid_argument("unknown NMR trial");
    return v;
}
struct Result {
    I profit = 0;
    Certificate certificate;
    std::vector<I> gains;
    size_t pairs = 0;
    double trace_ms = 0, original_dual_ms = 0, grouped_dual_ms = 0;
};
template <class Policy>
Result run_sweep(const Input &in, const Variant &v, const std::string &phase) {
    Result out;
    if (phase == "gains") {
        out.gains = gains<Policy>(in, v.stride);
        return out;
    }
    auto g = events<true>(in); // ordinary path below omits this mapping.
    if (phase == "dual" || phase == "dual-compare") {
        auto begin = std::chrono::steady_clock::now();
        auto result = traced<Policy>(in, g.events, v.stride);
        if (phase == "dual-compare") {
            out.profit = result.profit;
            auto trace_end = std::chrono::steady_clock::now();
            auto original = recover(in, g, result.matches, result.profit, false);
            auto original_end = std::chrono::steady_clock::now();
            out.certificate = recover(in, g, result.matches, result.profit, true);
            auto grouped_end = std::chrono::steady_clock::now();
            out.trace_ms = std::chrono::duration<double, std::milli>(trace_end - begin).count();
            out.original_dual_ms =
                std::chrono::duration<double, std::milli>(original_end - trace_end).count();
            out.grouped_dual_ms =
                std::chrono::duration<double, std::milli>(grouped_end - original_end).count();
            return out;
        }
        out.profit = result.profit;
        out.certificate = recover(in, g, result.matches, result.profit, v.grouped_dual);
    } else
        throw std::invalid_argument("invalid internal phase");
    return out;
}
inline Result run(const Input &in, const Variant &v, std::string phase,
                  size_t pair_limit = 8000000) {
    if (v.quadratic && in.p != 2)
        throw std::invalid_argument("quadratic requires p=2");
    if (v.costflow) {
        auto f = flow(in, phase == "gains" || phase == "both", phase == "dual", pair_limit);
        return {f.profit, std::move(f.certificate), std::move(f.gains), f.pairs};
    }
    if (phase == "both") {
        auto out = run(in, v, "solve", pair_limit);
        out.gains = run(in, v, "gains", pair_limit).gains;
        return out;
    }
    if (phase == "solve") {
        auto g = events<false>(in);
        I profit;
        if (v.quadratic)
            profit = ordinary<k::QuadraticPolicy>(in, g.events);
        else if (v.virt)
            profit = ordinary<k::VirtualPolicy>(in, g.events);
        else
            profit = convex_sweep::sweep_solve(
                g.events, [&](double a, double b) { return in.tau - in.model.cost(a, b); },
                in.model.radius(in.tau));
        return {profit};
    }
    if (v.quadratic)
        return run_sweep<k::QuadraticPolicy>(in, v, phase);
    if (v.virt)
        return run_sweep<k::VirtualPolicy>(in, v, phase);
    if (v.name != "baseline")
        return run_sweep<k::MaterialPolicy>(in, v, phase);
    // Baseline is production, not the fork, for both numerical and timing parity.
    Result out;
    if (phase == "dual") {
        auto g = events<true>(in);
        auto r = convex_sweep::sweep_solve<true>(
            g.events, [&](double a, double b) { return in.tau - in.model.cost(a, b); },
            in.model.radius(in.tau));
        out.profit = r.profit;
        out.certificate = recover(in, g, r.matches, r.profit, false);
        return out;
    }
    if (phase != "gains")
        throw std::invalid_argument("unknown phase");
    out.gains.resize(in.pos.size());
    for (bool mirror : {false, true}) {
        auto g = events<true>(in, mirror, true);
        std::vector<convex_sweep::Snapshot> snaps;
        std::vector<I> comp;
        auto profit = [&](double a, double b) { return in.tau - in.model.cost(a, b); };
        I base = convex_sweep::forward_store(g.events, profit, in.model.radius(in.tau), snaps);
        convex_sweep::adjoint_compose(g.events, profit, snaps, comp);
        for (size_t i = 0; i < g.events.size(); ++i)
            if (g.events[i].side == 1)
                for (auto part : g.parts[i])
                    out.gains[part.index] = std::max(out.gains[part.index], comp[i] - base);
    }
    return out;
}
} // namespace nmr_trials
#endif
