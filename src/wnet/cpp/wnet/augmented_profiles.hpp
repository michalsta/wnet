#ifndef WNET_AUGMENTED_PROFILES_HPP
#define WNET_AUGMENTED_PROFILES_HPP
#include "augmented_slope_dp.hpp"

namespace breakpoint_storage {
template<class Item, template<bool> class Tree = AugmentedTreap>
class AugmentedVertices {
    Tree<false> tree;
    // New profiles collect their output in a contiguous builder, then build
    // the treap in linear time. Existing immutable roots share a traversal
    // cache; structural updates invalidate only the updated profile's cache.
    bool building = true;
    std::vector<Item> buffer;
    mutable std::shared_ptr<std::vector<Item>> cache;
    void ready() {
        if (!building) return;
        std::vector<typename Tree<false>::Entry> entries; entries.reserve(buffer.size());
        for (auto item : buffer) entries.push_back({item.k,item.v});
        tree.build(entries); std::vector<Item>().swap(buffer); building = false; cache.reset();
    }
    const std::vector<Item>& values() const {
        if (building) return buffer;
        if (!cache) {
            auto entries = tree.entries();
            cache = std::make_shared<std::vector<Item>>(); cache->reserve(entries.size());
            for (auto item : entries) cache->push_back({item.key,item.value});
        }
        return *cache;
    }
    static Item item(typename Tree<false>::Entry e) { return {e.key,e.value}; }
    void normalize_ends() {
        // Structural transforms can create collinearity only at their joins.
        while (tree.size() >= 3) {
            const auto a = tree.first();
            const auto b = tree.neighbors(a.key).second;
            const auto c = tree.neighbors(b.key).second;
            if (!wide_int::products_equal(b.value-a.value,c.key-b.key,
                                         c.value-b.value,b.key-a.key)) break;
            tree.erase(b.key);
        }
        while (tree.size() >= 3) {
            const auto c = tree.last();
            const auto b = tree.neighbors(c.key-1).first;
            const auto a = tree.neighbors(b.key-1).first;
            if (!wide_int::products_equal(b.value-a.value,c.key-b.key,
                                         c.value-b.value,b.key-a.key)) break;
            tree.erase(b.key);
        }
    }
public:
    using const_iterator = typename std::vector<Item>::const_iterator;
    using iterator = const_iterator;
    bool empty() const { return building ? buffer.empty() : tree.empty(); }
    size_t size() const { return building ? buffer.size() : tree.size(); }
    Item front() const { return building ? buffer.front() : item(tree.first()); }
    Item back() const { return building ? buffer.back() : item(tree.last()); }
    auto begin() const { return values().begin(); }
    auto end() const { return values().end(); }
    auto rbegin() const { return values().rbegin(); }
    auto rend() const { return values().rend(); }
    auto lower_bound(int64_t k) const {
        const auto& data = values();
        return std::lower_bound(data.begin(),data.end(),k,[](auto e,auto key) { return e.k < key; });
    }
    void reserve(size_t n) { if (building) buffer.reserve(n); }
    void sort() { if (building) std::sort(buffer.begin(),buffer.end(),[](auto a,auto b) { return a.k < b.k; }); }
    void push_back(Item e) {
        if (building) buffer.push_back(e);
        else {
            const auto [a,b] = tree.neighbors(e.k);
            tree.insert(e.k,a.key == e.k ? std::max(a.value,e.v) : e.v); cache.reset();
        }
    }
    void insert(const_iterator,Item e) { ready(); push_back(e); }
    AugmentedVertices& operator=(std::vector<Item>&& entries) {
        buffer = std::move(entries); building = true; ready(); return *this;
    }
    int64_t eval(int64_t k) const {
        if (k <= front().k) return front().v;
        if (k >= back().k) return back().v;
        if (building) {
            auto b = lower_bound(k); const auto a = std::prev(b);
            return a->v + wide_int::mul_div_trunc(b->v-a->v,k-a->k,b->k-a->k);
        }
        const auto [a,b] = tree.neighbors(k);
        if (a.key == k) return a.value;
        return a.value + wide_int::mul_div_trunc(b.value-a.value,k-a.key,b.key-a.key);
    }
    std::pair<int64_t,int64_t> vertex_argmax(int64_t lo,int64_t hi) const {
        if (!building) return tree.vertex_max(lo,hi);
        std::pair<int64_t,int64_t> best{lo,std::numeric_limits<int64_t>::min()};
        for (auto e : buffer) if (lo <= e.k && e.k <= hi && e.v > best.second) best = {e.k,e.v};
        return best;
    }
    void pointwise_max(const AugmentedVertices& other) {
        const auto& a = values(); const auto& b = other.values();
        std::vector<int64_t> keys; keys.reserve(a.size()+b.size());
        size_t ia = 0,ib = 0;
        while (ia < a.size() || ib < b.size()) {
            int64_t k;
            if (ib == b.size() || (ia < a.size() && a[ia].k < b[ib].k)) k = a[ia++].k;
            else if (ia == a.size() || b[ib].k < a[ia].k) k = b[ib++].k;
            else { k = a[ia++].k; ++ib; }
            keys.push_back(k);
        }
        constexpr int64_t unreachable = std::numeric_limits<int64_t>::min()/4;
        auto value = [&](const auto& points,size_t cursor,int64_t k) {
            if (k > points.back().k) return unreachable;
            if (cursor+1 == points.size()) return points.back().v;
            const auto x = points[cursor],y = points[cursor+1];
            return x.v + wide_int::mul_div_trunc(y.v-x.v,k-x.k,y.k-x.k);
        };
        std::vector<Item> out; out.reserve(3*keys.size());
        auto append = [&](int64_t k,int64_t v) {
            if (!out.empty() && out.back().k == k) { out.back().v = std::max(out.back().v,v); return; }
            while (out.size() >= 2) {
                const auto x = out[out.size()-2],y = out.back();
                if (!wide_int::products_equal(y.v-x.v,k-y.k,v-y.v,y.k-x.k)) break;
                out.pop_back();
            }
            out.push_back({k,v});
        };
        ia = ib = 0;
        for (size_t i = 0; i < keys.size(); ++i) {
            const auto k = keys[i];
            while (ia+1 < a.size() && a[ia+1].k <= k) ++ia;
            while (ib+1 < b.size() && b[ib+1].k <= k) ++ib;
            auto fa = [&](int64_t x) { return value(a,ia,x); };
            auto fb = [&](int64_t x) { return value(b,ib,x); };
            append(k,std::max(fa(k),fb(k)));
            if (i+1 == keys.size() || keys[i+1]-k <= 1) continue;
            const int64_t end = keys[i+1],d1 = fa(k)-fb(k),d2 = fa(end)-fb(end);
            if ((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) {
                int64_t lo = k,hi = end;
                while (hi-lo > 1) {
                    const auto mid = lo+(hi-lo)/2,d = fa(mid)-fb(mid);
                    if (d1 > 0 ? d > 0 : d < 0) lo = mid; else hi = mid;
                }
                append(lo,std::max(fa(lo),fb(lo))); append(hi,std::max(fa(hi),fb(hi)));
            }
        }
        *this = std::move(out);
    }
    void set_front_value(int64_t value) { ready(); tree.insert(tree.first().key,value); cache.reset(); }
    void truncate(int64_t k) {
        const auto value = eval(k); ready();
        auto [keep,discard] = std::move(tree).split(k);
        tree = std::move(keep); tree.insert(k,value); normalize_ends(); cache.reset();
    }
    void shift_append(int64_t q) {
        const auto value = front().v; ready(); tree.shift_keys(q); tree.insert(0,value);
        normalize_ends(); cache.reset();
    }
    AugmentedVertices reflected(int64_t domain) const {
        auto out = *this; out.ready(); out.tree.reflect(domain); out.cache.reset(); return out;
    }
    void extend_flat(int64_t k) {
        const auto value = back().v; ready(); tree.insert(k,value); normalize_ends(); cache.reset();
    }
    void shift_left_clamp(int64_t q) {
        const auto domain = back().k;
        q = std::min(q,domain);
        const auto value = eval(q); ready();
        auto [discard,keep] = std::move(tree).split(q);
        tree = std::move(keep); tree.insert(q,value); tree.shift_keys(-q);
        if (q) tree.insert(domain,value_at_last());
        normalize_ends(); cache.reset();
    }
    int64_t value_at_last() const { return tree.last().value; }
};

template<template<bool> class Tree = AugmentedTreap>
struct AugmentedStorage {
    template<class Item> using Vertices = AugmentedVertices<Item,Tree>;
    template<class V> using Slope = slopedp_detail::AugmentedConvexPL<V,Tree>;
};
}  // namespace breakpoint_storage
#endif
