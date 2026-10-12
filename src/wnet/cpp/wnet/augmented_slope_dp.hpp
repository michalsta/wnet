#ifndef WNET_AUGMENTED_SLOPE_DP_HPP
#define WNET_AUGMENTED_SLOPE_DP_HPP
#include "augmented_treap.hpp"

namespace slopedp_detail {
template<class V, template<bool> class Tree = breakpoint_storage::AugmentedTreap>
struct AugmentedConvexPL {
    using W = wide_int::wide_t;
    static constexpr V NEG_BIG = std::numeric_limits<V>::min()/4;
    static constexpr V POS_BIG = std::numeric_limits<V>::max()/4;
    Tree<true> L,R;
    V offL = 0,offR = 0;
    W minval = 0;
    V maxLeft() const { return L.empty() ? NEG_BIG : V(L.last().key+offL); }
    V minRight() const { return R.empty() ? POS_BIG : V(R.first().key+offR); }
    void insL(V pos,V mass) { if (mass) L.insert(pos-offL,mass,true); }
    void insR(V pos,V mass) { if (mass) R.insert(pos-offR,mass,true); }
    void addAbs(V b,V gamma) {
        if (gamma <= 0) return;
        const V ml = maxLeft(),mr = minRight();
        if (b >= ml && b <= mr) { insL(b,gamma); insR(b,gamma); return; }
        V take;
        if (b < ml) {
            auto [low,eligible] = std::move(L).split(b-offL,true);
            take = std::min<V>(gamma,eligible.total_mass());
            auto [keep,chunk] = std::move(eligible).split_by_mass(eligible.total_mass()-take);
            minval += chunk.moment() + W(offL-b)*take;
            L = Tree<true>::join(std::move(low),std::move(keep));
            chunk.shift_keys(offL-offR);
            R = Tree<true>::join(std::move(chunk),std::move(R));
            insL(b,gamma+take); insR(b,gamma-take);
        } else {
            auto [eligible,high] = std::move(R).split(b-offR);
            take = std::min<V>(gamma,eligible.total_mass());
            auto [chunk,keep] = std::move(eligible).split_by_mass(take);
            minval += W(b-offR)*take - chunk.moment();
            R = Tree<true>::join(std::move(keep),std::move(high));
            chunk.shift_keys(offR-offL);
            L = Tree<true>::join(std::move(L),std::move(chunk));
            insR(b,gamma+take); insL(b,gamma-take);
        }
    }
    std::pair<V,V> convV(V tau,V v,V t) {
        V hi = POS_BIG;
        if (R.total_mass() >= tau) {
            auto [keep,discard] = std::move(R).split_by_mass(tau);
            R = std::move(keep);
            hi = R.empty() ? POS_BIG : V(R.last().key+offR);
        }
        const V lo = maxLeft(); offL -= t; offR += v;
        return {lo,hi};
    }
    W evalAt(V s) const {
        W value = minval;
        if (s < maxLeft()) {
            const auto prefix = L.less(s-offL,true);
            const V mass = L.total_mass()-prefix.mass;
            value += L.moment()-prefix.moment + W(offL-s)*mass;
        } else if (s > minRight()) {
            const auto prefix = R.less(s-offR);
            value += W(s-offR)*prefix.mass-prefix.moment;
        }
        return value;
    }
};
}  // namespace slopedp_detail
#endif
