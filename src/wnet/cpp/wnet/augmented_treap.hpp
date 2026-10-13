#ifndef WNET_AUGMENTED_TREAP_HPP
#define WNET_AUGMENTED_TREAP_HPP

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
#include "wide_int.hpp"

namespace breakpoint_storage {

// Persistent split/join treap. Mass=true augments positive slope masses with
// their total and first moment. Mass=false augments profile vertices with
// maximum value and its first/last coordinates. Root copies are O(1); only
// modified paths are copied. Key transforms and profile value shifts are lazy.
template<bool Mass>
class AugmentedTreap {
public:
    using I = int64_t;
    using W = wide_int::wide_t;
    struct Entry { I key, value; };
    struct Totals { I mass = 0; W moment = 0; };
private:
    struct Node;
    using Ptr = std::shared_ptr<Node>;
    struct MassAggregate { I mass = 0; W moment = 0; };
    struct ValueAggregate { I maximum = 0,max_first = 0,max_last = 0; };
    struct Node : std::conditional_t<Mass,MassAggregate,ValueAggregate> {
        I key, value;
        uint64_t priority;
        Ptr left, right;
        size_t count = 1;
        Entry first, last;
        int key_sign = 1;
        I key_shift = 0, value_shift = 0;
        Node(I k, I v, uint64_t p) : key(k), value(v), priority(p),
            first{k,v}, last{k,v} {
            if constexpr (Mass) { this->mass = v; this->moment = W(v)*k; }
            else { this->maximum = v; this->max_first = this->max_last = k; }
        }
    };
    Ptr root;
    static uint64_t priority() {
        static thread_local uint64_t sequence = 0x9e3779b97f4a7c15ULL;
        uint64_t z = (sequence += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
    static size_t count(const Ptr& p) { return p ? p->count : 0; }
    static I checked_sum(I a, I b) {
        if (b > 0 && a > std::numeric_limits<I>::max() - b)
            throw std::overflow_error("Augmented treap mass overflow");
        return a+b;
    }
    static void own(Ptr& p) { if (p && p.use_count() != 1) p = std::make_shared<Node>(*p); }
    static I transform(I k, int sign, I shift) { return sign == 1 ? k+shift : shift-k; }
    static void apply(Ptr& p, int sign, I shift, I dv = 0) {
        if (!p) return;
        own(p);
        p->key = transform(p->key, sign, shift); p->value += dv;
        auto first = p->first, last = p->last;
        if (sign == -1) { std::swap(first,last); std::swap(p->left,p->right); }
        p->first = {transform(first.key,sign,shift), first.value+dv};
        p->last = {transform(last.key,sign,shift), last.value+dv};
        if constexpr (Mass) {
            if (dv) throw std::invalid_argument("Cannot shift slope masses");
            p->moment = (sign == 1 ? p->moment : -p->moment) + W(p->mass)*shift;
        } else {
            const I a = p->max_first, b = p->max_last;
            p->maximum += dv;
            p->max_first = transform(sign == 1 ? a : b,sign,shift);
            p->max_last = transform(sign == 1 ? b : a,sign,shift);
        }
        p->key_sign *= sign;
        p->key_shift = transform(p->key_shift,sign,shift);
        p->value_shift += dv;
    }
    static void push(Ptr& p) {
        own(p);
        if (p->key_sign != 1 || p->key_shift || p->value_shift) {
            apply(p->left,p->key_sign,p->key_shift,p->value_shift);
            apply(p->right,p->key_sign,p->key_shift,p->value_shift);
            p->key_sign = 1; p->key_shift = p->value_shift = 0;
        }
    }
    static void pull(Ptr& p) {
        p->count = 1 + count(p->left) + count(p->right);
        p->first = p->left ? p->left->first : Entry{p->key,p->value};
        p->last = p->right ? p->right->last : Entry{p->key,p->value};
        if constexpr (Mass) {
            p->mass = p->value;
            p->moment = W(p->value)*p->key;
            auto include = [&](const Ptr& child) {
                if (!child) return;
                p->mass = checked_sum(p->mass,child->mass);
                p->moment += child->moment;
            };
            include(p->left); include(p->right);
        } else {
            p->maximum = p->value; p->max_first = p->max_last = p->key;
            auto include = [&](const Ptr& child) {
                if (!child) return;
                if (child->maximum > p->maximum) {
                    p->maximum = child->maximum;
                    p->max_first = child->max_first; p->max_last = child->max_last;
                } else if (child->maximum == p->maximum) {
                    p->max_first = std::min(p->max_first,child->max_first);
                    p->max_last = std::max(p->max_last,child->max_last);
                }
            };
            include(p->left); include(p->right);
        }
    }
    static Ptr merge(Ptr a, Ptr b) {
        if (!a) return b;
        if (!b) return a;
        if (a->priority >= b->priority) {
            push(a); a->right = merge(std::move(a->right),std::move(b)); pull(a); return a;
        }
        push(b); b->left = merge(std::move(a),std::move(b->left)); pull(b); return b;
    }
    static std::pair<Ptr,Ptr> split_key(Ptr p, I key, bool inclusive) {
        if (!p) return {};
        push(p);
        if (p->key < key || (inclusive && p->key == key)) {
            auto [a,b] = split_key(std::move(p->right),key,inclusive);
            p->right = std::move(a); pull(p); return {std::move(p),std::move(b)};
        }
        auto [a,b] = split_key(std::move(p->left),key,inclusive);
        p->left = std::move(b); pull(p); return {std::move(a),std::move(p)};
    }
    static std::pair<Ptr,Ptr> split_mass(Ptr p, I mass) requires Mass {
        if (!p) return {};
        if (mass <= 0) return {{},std::move(p)};
        if (mass >= p->mass) return {std::move(p),{}};
        push(p);
        const I left_mass = p->left ? p->left->mass : 0;
        if (mass < left_mass) {
            auto [a,b] = split_mass(std::move(p->left),mass);
            // A new boundary fragment can have higher priority than p.
            // Merge it back to preserve heap order along the split path.
            pull(p); return {std::move(a),merge(std::move(b),std::move(p))};
        }
        if (mass > left_mass + p->value) {
            auto [a,b] = split_mass(std::move(p->right),mass-left_mass-p->value);
            p->right = std::move(a); pull(p); return {std::move(p),std::move(b)};
        }
        const I take = mass-left_mass,original = p->value;
        if (!take) {
            auto a = std::move(p->left); pull(p); return {std::move(a),std::move(p)};
        }
        if (take == original) {
            auto b = std::move(p->right); pull(p); return {std::move(p),std::move(b)};
        }
        // The new half needs an independent priority. Copying the old
        // priority into both halves creates equal-priority chains after
        // chunks move between trees with different lazy coordinate offsets.
        auto right = merge(std::make_shared<Node>(p->key,original-take,priority()),
                           std::move(p->right));
        p->value = take; pull(p);
        return {std::move(p),std::move(right)};
    }
    static void walk(const Ptr& p, int sign, I shift, I dv, std::vector<Entry>& out) {
        if (!p) return;
        const int next_sign = sign*p->key_sign;
        const I next_shift = transform(p->key_shift,sign,shift), next_dv = dv+p->value_shift;
        const auto& left = sign == 1 ? p->left : p->right;
        const auto& right = sign == 1 ? p->right : p->left;
        walk(left,next_sign,next_shift,next_dv,out);
        out.push_back({transform(p->key,sign,shift),p->value+dv});
        walk(right,next_sign,next_shift,next_dv,out);
    }
    static void aggregate_less(const Ptr& p, I key, bool inclusive,
                               int sign, I shift, Totals& out) requires Mass {
        if (!p) return;
        const I lo = transform((sign == 1 ? p->first : p->last).key,sign,shift);
        const I hi = transform((sign == 1 ? p->last : p->first).key,sign,shift);
        if (lo > key || (!inclusive && lo == key)) return;
        if (hi < key || (inclusive && hi == key)) {
            out.mass = checked_sum(out.mass,p->mass);
            out.moment += (sign == 1 ? p->moment : -p->moment) + W(p->mass)*shift;
            return;
        }
        const int ns = sign*p->key_sign;
        const I sh = transform(p->key_shift,sign,shift);
        aggregate_less(sign == 1 ? p->left : p->right,key,inclusive,ns,sh,out);
        const I k = transform(p->key,sign,shift);
        if (k < key || (inclusive && k == key)) {
            out.mass = checked_sum(out.mass,p->value); out.moment += W(p->value)*k;
            aggregate_less(sign == 1 ? p->right : p->left,key,inclusive,ns,sh,out);
        }
    }
    static void maximum_in(const Ptr& p,I lo,I hi,int sign,I shift,I dv,
                           std::pair<I,I>& best) requires (!Mass) {
        if (!p) return;
        const I first = transform((sign == 1 ? p->first : p->last).key,sign,shift);
        const I last = transform((sign == 1 ? p->last : p->first).key,sign,shift);
        if (first > hi || last < lo) return;
        auto consider = [&](I k,I v) {
            if (v > best.second || (v == best.second && k < best.first)) best = {k,v};
        };
        if (first >= lo && last <= hi) {
            consider(transform(sign == 1 ? p->max_first : p->max_last,sign,shift),p->maximum+dv);
            return;
        }
        const int ns = sign*p->key_sign;
        const I sh = transform(p->key_shift,sign,shift), nd = dv+p->value_shift;
        maximum_in(sign == 1 ? p->left : p->right,lo,hi,ns,sh,nd,best);
        const I k = transform(p->key,sign,shift);
        if (lo <= k && k <= hi) consider(k,p->value+dv);
        maximum_in(sign == 1 ? p->right : p->left,lo,hi,ns,sh,nd,best);
    }
public:
    bool empty() const { return !root; }
    size_t size() const { return count(root); }
    Entry first() const { return root->first; }
    Entry last() const { return root->last; }
    I total_mass() const requires Mass { return root ? root->mass : 0; }
    W moment() const requires Mass { return root ? root->moment : W(0); }
    void shift_keys(I delta) { apply(root,1,delta); }
    void shift_values(I delta) requires (!Mass) { apply(root,1,0,delta); }
    void reflect(I origin) { apply(root,-1,origin); }
    size_t height() const {
        size_t maximum = 0;
        std::vector<std::pair<const Node*,size_t>> stack;
        if (root) stack.push_back({root.get(),1});
        while (!stack.empty()) {
            auto [p,depth] = stack.back(); stack.pop_back(); maximum = std::max(maximum,depth);
            if (p->left) stack.push_back({p->left.get(),depth+1});
            if (p->right) stack.push_back({p->right.get(),depth+1});
        }
        return maximum;
    }
    bool heap_ordered() const {
        std::vector<const Node*> stack;
        if (root) stack.push_back(root.get());
        while (!stack.empty()) {
            const auto p = stack.back(); stack.pop_back();
            for (const auto child : {p->left.get(),p->right.get()}) {
                if (!child) continue;
                if (child->priority > p->priority) return false;
                stack.push_back(child);
            }
        }
        return true;
    }
    std::vector<Entry> entries() const {
        std::vector<Entry> out; out.reserve(size()); walk(root,1,0,0,out); return out;
    }
    // O(n) Cartesian-tree construction from strictly increasing keys.
    void build(const std::vector<Entry>& entries) {
        std::vector<Ptr> stack; stack.reserve(entries.size());
        for (auto e : entries) {
            auto p = std::make_shared<Node>(e.key,e.value,priority());
            Ptr left;
            while (!stack.empty() && stack.back()->priority < p->priority) {
                left = std::move(stack.back()); stack.pop_back(); pull(left);
            }
            p->left = std::move(left);
            if (!stack.empty()) stack.back()->right = p;
            stack.push_back(std::move(p));
        }
        while (stack.size() > 1) { pull(stack.back()); stack.pop_back(); }
        if (!stack.empty()) pull(stack.front());
        root = stack.empty() ? Ptr{} : std::move(stack.front());
    }
    std::pair<AugmentedTreap,AugmentedTreap> split(I key,bool inclusive = false) && {
        auto [a,b] = split_key(std::move(root),key,inclusive);
        AugmentedTreap left,right; left.root = std::move(a); right.root = std::move(b);
        return {std::move(left),std::move(right)};
    }
    std::pair<AugmentedTreap,AugmentedTreap> split_by_mass(I mass) && requires Mass {
        auto [a,b] = split_mass(std::move(root),mass);
        AugmentedTreap left,right; left.root = std::move(a); right.root = std::move(b);
        return {std::move(left),std::move(right)};
    }
    static AugmentedTreap join(AugmentedTreap a,AugmentedTreap b) {
        if (!a.empty() && !b.empty()) {
            if (a.last().key > b.first().key) throw std::runtime_error("Unordered treap join");
            if (a.last().key == b.first().key) {
                const I key = a.last().key;
                auto [al,ar] = std::move(a).split(key);
                auto [bl,br] = std::move(b).split(key,true);
                const I value = Mass ? checked_sum(ar.first().value,bl.first().value)
                                     : std::max(ar.first().value,bl.first().value);
                push(ar.root); ar.root->value = value; pull(ar.root);
                al.root = merge(std::move(al.root),std::move(ar.root));
                al.root = merge(std::move(al.root),std::move(br.root));
                return al;
            }
        }
        a.root = merge(std::move(a.root),std::move(b.root)); return a;
    }
    void insert(I key,I value,bool accumulate = false) {
        if constexpr (Mass) if (value < 0) throw std::invalid_argument("Negative treap mass");
        auto [a,rest] = split_key(std::move(root),key,false);
        auto [equal,b] = split_key(std::move(rest),key,true);
        if (equal) {
            push(equal);
            equal->value = accumulate ? checked_sum(equal->value,value) : value;
            pull(equal);
        } else equal = std::make_shared<Node>(key,value,priority());
        root = merge(merge(std::move(a),std::move(equal)),std::move(b));
    }
    void erase(I key) {
        auto [a,rest] = split_key(std::move(root),key,false);
        auto [equal,b] = split_key(std::move(rest),key,true);
        root = merge(std::move(a),std::move(b));
    }
    Totals less(I key,bool inclusive = false) const requires Mass {
        Totals out; aggregate_less(root,key,inclusive,1,0,out); return out;
    }
    std::pair<I,I> vertex_max(I lo,I hi) const requires (!Mass) {
        std::pair<I,I> best{lo,std::numeric_limits<I>::min()};
        maximum_in(root,lo,hi,1,0,0,best); return best;
    }
    // Exact predecessor/successor without pushing or allocating tree nodes.
    std::pair<Entry,Entry> neighbors(I key) const {
        Entry before = first(),after = last();
        const Node* p = root.get(); int sign = 1; I shift = 0,dv = 0;
        while (p) {
            const I k = transform(p->key,sign,shift);
            const Entry e{k,p->value+dv};
            const int ns = sign*p->key_sign;
            const I sh = transform(p->key_shift,sign,shift), nd = dv+p->value_shift;
            if (k <= key) { before = e; p = (sign == 1 ? p->right : p->left).get(); }
            else { after = e; p = (sign == 1 ? p->left : p->right).get(); }
            sign = ns; shift = sh; dv = nd;
        }
        return {before,after};
    }
};
}  // namespace breakpoint_storage
#endif
