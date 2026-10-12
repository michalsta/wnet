// C++20 -O3, with wnet/cpp and installed pylmcf/cpp include paths.
// Args: positions radius storage(flat/tree/augmented) mode repetitions
// Modes: slope, cost, trace, duals, marginals. Each process measures one
// variant so /usr/bin/time can attribute peak RSS independently.
#include "wnet/decompositable_graph.hpp"
#include <chrono>
#include <cstdio>
#include <numeric>
#include <random>

using Clock = std::chrono::steady_clock;
using I = int64_t;
struct Input {
    std::vector<I> e,t,ce,ct;
    std::vector<double> positions;
    std::vector<convex_sweep::Event> events;
    I tau;
    double radius;
};

template<class DP>
I slope(const Input& in,I reward) {
    DP dp;
    const I wall = std::numeric_limits<I>::max()/1024;
    dp.insL(0,wall); dp.insR(0,wall);
    for (size_t i = 0; i < in.e.size(); ++i) {
        dp.convV(reward,in.e[i]-in.t[i],in.t[i]);
        dp.minval += -wide_int::wide_t(reward)*in.t[i];
        if (i+1 < in.e.size()) dp.addAbs(0,1);
    }
    return I(dp.evalAt(0));
}

template<class Profile>
I sweep(const Input& in,const std::string& mode) {
    auto cost = [](double a,double b) -> I { return std::llround((a-b)*(a-b)); };
    auto profit = [&](double a,double b) { return in.tau-cost(a,b); };
    if (mode == "cost") return convex_sweep::sweep_solve<false,Profile>(in.events,profit,in.radius);
    if (mode == "trace" || mode == "duals") {
        auto result = convex_sweep::sweep_solve<true,Profile>(in.events,profit,in.radius);
        if (mode == "duals")
            convex_sweep_dual::recover<I>(in.positions,in.ce,in.ct,in.tau,in.radius,cost,
                                          result.matches,wide_int::wide_t(result.profit));
        return result.profit;
    }
    if (mode != "marginals") throw std::invalid_argument("Unknown benchmark mode");
    std::vector<I> gains(in.positions.size(),0);
    for (bool mirror : {false,true}) {
        auto events = in.events;
        if (mirror) {
            std::reverse(events.begin(),events.end());
            for (auto& event : events) event.pos = -event.pos;
        }
        std::vector<convex_sweep::BasicSnapshot<Profile>> snapshots;
        const I opt = convex_sweep::forward_store(events,profit,in.radius,snapshots);
        std::vector<I> composed;
        convex_sweep::adjoint_compose(events,profit,snapshots,composed);
        for (size_t i = 0; i < events.size(); ++i) if (events[i].side == 1)
            gains[events[i].index] = std::max(gains[events[i].index],composed[i]-opt);
    }
    return std::accumulate(gains.begin(),gains.end(),I(0));
}

int main(int argc,char** argv) {
    const size_t n = argc > 1 ? std::stoull(argv[1]) : 1000;
    const double radius = argc > 2 ? std::stod(argv[2]) : 8;
    const std::string storage = argc > 3 ? argv[3] : "augmented";
    const std::string mode = argc > 4 ? argv[4] : "cost";
    const size_t repetitions = argc > 5 ? std::stoull(argv[5]) : 5;
    if (!n || radius <= 0 || !repetitions) throw std::invalid_argument("Invalid benchmark dimensions");
    Input in;
    in.tau = I(radius*radius); in.radius = std::sqrt(double(in.tau)+1);
    in.e.resize(n); in.t.resize(n); in.ce.resize(2*n); in.ct.resize(2*n);
    in.positions.resize(2*n);
    std::mt19937 rng(42);
    for (size_t i = 0; i < n; ++i) {
        in.e[i] = 1+rng()%7; in.t[i] = 1+rng()%7;
        in.positions[2*i] = double(i); in.positions[2*i+1] = double(i)+0.25;
        in.ce[2*i] = in.e[i]; in.ct[2*i+1] = in.t[i];
        in.events.push_back({double(i),0,in.e[i],2*i});
        in.events.push_back({double(i)+0.25,1,in.t[i],2*i+1});
    }
    auto run = [&]() -> I {
        if (mode == "slope") {
            if (storage == "flat") return slope<slopedp_detail::ConvexPL<I>>(in,I(radius));
            if (storage == "tree") return slope<slopedp_detail::ConvexPL<I,breakpoint_storage::MapStorage>>(in,I(radius));
            if (storage == "augmented") return slope<slopedp_detail::AugmentedConvexPL<I>>(in,I(radius));
        } else {
            if (storage == "flat") return sweep<convex_sweep::PLF>(in,mode);
            if (storage == "tree") return sweep<convex_sweep::TreePLF>(in,mode);
            if (storage == "augmented") return sweep<convex_sweep::AugmentedPLF>(in,mode);
        }
        throw std::invalid_argument("Unknown benchmark storage");
    };
    const I expected = run(); // Warmup, outside timed samples.
    std::vector<double> samples;
    for (size_t i = 0; i < repetitions; ++i) {
        const auto start = Clock::now(); const I value = run();
        samples.push_back(std::chrono::duration<double,std::milli>(Clock::now()-start).count());
        if (value != expected) throw std::runtime_error("Benchmark objective changed");
    }
    std::sort(samples.begin(),samples.end());
    std::printf("{\"n\":%zu,\"radius\":%.9g,\"storage\":\"%s\",\"mode\":\"%s\",\"repetitions\":%zu,\"median_ms\":%.6f,\"min_ms\":%.6f,\"max_ms\":%.6f,\"checksum\":%lld}\n",
        n,radius,storage.c_str(),mode.c_str(),repetitions,samples[samples.size()/2],samples.front(),samples.back(),(long long)expected);
}
