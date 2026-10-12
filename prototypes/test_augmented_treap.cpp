// C++20, -Isrc/wnet/cpp/wnet; optionally software int128 and UBSan.
#include "convex_sweep.hpp"
#include <cassert>
#include <map>
#include <random>

using I = int64_t;
using W = wide_int::wide_t;
using Mass = breakpoint_storage::AugmentedTreap<true>;
using Values = breakpoint_storage::AugmentedTreap<false>;

int main() {
    std::mt19937 rng(914);
    for (size_t trial = 0; trial < 1000; ++trial) {
        Mass tree;
        std::map<I,I> oracle;
        for (size_t step = 0; step < 60; ++step) {
            const I key = I(rng()%101)-50, amount = 1+rng()%20;
            tree.insert(key,amount,true); oracle[key] += amount;
            I total = 0; W moment = 0;
            for (auto [k,m] : oracle) { total += m; moment += W(k)*m; }
            assert(tree.total_mass() == total && tree.moment() == moment);
            const auto snapshot = tree;
            const auto original = tree.entries();
            const I cut = rng()%(total+1);
            auto [a,b] = std::move(tree).split_by_mass(cut);
            assert(a.total_mass() == cut && b.total_mass() == total-cut);
            assert(a.heap_ordered() && b.heap_ordered());
            tree = Mass::join(std::move(a),std::move(b));
            assert(tree.heap_ordered());
            assert(tree.moment() == moment && snapshot.moment() == moment);
            const auto entries = snapshot.entries();
            assert(entries.size() == original.size());
            const auto prefix = tree.less(key);
            I mass = 0; W sum = 0;
            for (auto [k,m] : oracle) if (k < key) { mass += m; sum += W(k)*m; }
            assert(prefix.mass == mass && prefix.moment == sum);
            const I shift = I(rng()%7)-3;
            tree.shift_keys(shift);
            assert(tree.moment() == moment+W(total)*shift);
            tree.shift_keys(-shift);
            tree.erase(key); oracle.erase(key);
        }
    }
    for (size_t trial = 0; trial < 1000; ++trial) {
        convex_sweep::PLF flat;
        flat.vs.push_back({0,I(rng()%31)-15});
        for (size_t i = 0; i < 8; ++i) {
            const I length = 1+rng()%8, slope = I(rng()%11)-5;
            auto last = flat.vs.back(); flat.vs.push_back({last.k+length,last.v+slope*length});
        }
        flat.normalize();
        convex_sweep::AugmentedPLF aug;
        std::vector<convex_sweep::AugmentedPLF::V> data;
        for (auto e : flat.vs) data.push_back({e.k,e.v});
        aug.vs = std::move(data);
        for (size_t step = 0; step < 20; ++step) {
            const auto saved = aug;
            const auto saved_flat = flat;
            const I q = 1+rng()%7;
            switch (rng()%6) {
                case 0: flat.shift_append(q); aug.shift_append(q); break;
                case 1: {
                    const I k = rng()%(flat.K()+1); flat.truncate(k); aug.truncate(k); break;
                }
                case 2: flat = convex_sweep::reflect_pl(flat); aug = convex_sweep::reflect_pl(aug); break;
                case 3: flat = convex_sweep::extend_flat(flat,flat.K()+q); aug = convex_sweep::extend_flat(aug,aug.K()+q); break;
                case 4: flat = convex_sweep::shift_left_clamp(flat,q); aug = convex_sweep::shift_left_clamp(aug,q); break;
                case 5: flat.suffix_max(); aug.suffix_max(); break;
            }
            assert(flat.K() == aug.K());
            for (I k = 0; k <= flat.K(); ++k) assert(flat.eval(k) == aug.eval(k));
            for (I k = 0; k <= saved_flat.K(); ++k) assert(saved.eval(k) == saved_flat.eval(k));
            const I lo = rng()%(flat.K()+1), hi = lo+rng()%(flat.K()-lo+1);
            assert(convex_sweep::argmax(flat,lo,hi).second == convex_sweep::argmax(aug,lo,hi).second);
        }
    }
}
