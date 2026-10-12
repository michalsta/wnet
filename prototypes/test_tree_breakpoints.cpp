// Compile with C++20, the wnet/cpp and installed pylmcf/cpp include paths.
// Optional -DWNET_FORCE_SOFT_INT128 -fsanitize=undefined checks portability.
#include "wnet/decompositable_graph.hpp"
#include <cassert>
#include <random>

static size_t tree_lookups = 0;
// A second map-compatible tree policy verifies that kernels are not tied to
// std::map's concrete type. Instrumentation needs no virtual dispatch.
template<class... Args> struct InstrumentedTree : std::map<Args...> {
    using Base = std::map<Args...>;
    auto lower_bound(const typename Base::key_type& key) {
        ++tree_lookups;
        return Base::lower_bound(key);
    }
    auto lower_bound(const typename Base::key_type& key) const {
        ++tree_lookups;
        return Base::lower_bound(key);
    }
};

int main() {
    using I = int64_t;
    using CustomStorage = breakpoint_storage::BalancedTreeStorage<InstrumentedTree>;
    std::mt19937 rng(42);
    for (size_t trial = 0; trial < 1000; ++trial) {
        slopedp_detail::ConvexPL<I> flat;
        slopedp_detail::ConvexPL<I, breakpoint_storage::MapStorage> map;
        slopedp_detail::ConvexPL<I, CustomStorage> custom;
        slopedp_detail::AugmentedConvexPL<I> augmented;
        const I wall = std::numeric_limits<I>::max() / 1024;
        flat.insL(0, wall); flat.insR(0, wall);
        map.insL(0, wall); map.insR(0, wall);
        custom.insL(0, wall); custom.insR(0, wall);
        augmented.insL(0,wall); augmented.insR(0,wall);
        const I tau = 1 + rng() % 100;
        for (size_t step = 0; step < 40; ++step) {
            const I e = rng() % 20, t = rng() % 20, gamma = rng() % 10;
            const auto interval = flat.convV(tau, e-t, t);
            assert(map.convV(tau, e-t, t) == interval);
            assert(custom.convV(tau, e-t, t) == interval);
            assert(augmented.convV(tau,e-t,t) == interval);
            flat.addAbs(0, gamma); map.addAbs(0, gamma); custom.addAbs(0, gamma); augmented.addAbs(0,gamma);
            for (I k : {-50, -1, 0, 1, 50}) {
                assert(map.evalAt(k) == flat.evalAt(k));
                assert(custom.evalAt(k) == flat.evalAt(k));
                assert(augmented.evalAt(k) == flat.evalAt(k));
            }
        }
    }
    // Repeated partial mass splits must not duplicate priorities. Moving
    // equal-priority fragments across offset trees used to create long chains.
    slopedp_detail::AugmentedConvexPL<I> stress;
    const I wall = std::numeric_limits<I>::max()/1024;
    stress.insL(0,wall); stress.insR(0,wall);
    for (size_t step = 0; step < 20000; ++step) {
        const I e = 1+rng()%7,t = 1+rng()%7;
        stress.convV(1000000,e-t,t); stress.addAbs(0,1);
    }
    assert(stress.L.height() < 128 && stress.R.height() < 128);
    using CustomProfile = convex_sweep::BasicPLF<CustomStorage>;
    std::vector<convex_sweep::Event> events{{0,0,3},{1,1,2},{2,0,1},{3,1,4}};
    auto profit = [](double a, double b) -> I { return 10 - I((a-b)*(a-b)); };
    const auto flat = convex_sweep::sweep_solve<true>(events, profit, 4);
    const auto custom = convex_sweep::sweep_solve<true, CustomProfile>(events, profit, 4);
    assert(flat.profit == custom.profit);
    assert(tree_lookups > 0);
}
