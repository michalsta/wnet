#ifndef WNET_ORDERED_BREAKPOINTS_HPP
#define WNET_ORDERED_BREAKPOINTS_HPP

#include <algorithm>
#include <deque>
#include <iterator>
#include <map>
#include <type_traits>
#include <utility>
#include <vector>

namespace breakpoint_storage {

// The tree policy is a map-like class template, supplied at compile time.
// It needs bidirectional ordered iterators, lower_bound, emplace_hint,
// insert and erase. Keys never change in place; shifts use solver offsets.
template<class Item, auto Key, template<class...> class Tree = std::map>
class OrderedTree {
    using K = std::remove_cvref_t<decltype(std::declval<Item>().*Key)>;
    Tree<K, Item> data;
    template<class It> class ValueIterator {
        It it;
        friend class OrderedTree;
    public:
        using iterator_category = std::bidirectional_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = Item;
        using reference = decltype((std::declval<It>()->second));
        using pointer = std::add_pointer_t<std::remove_reference_t<reference>>;
        ValueIterator() = default;
        explicit ValueIterator(It i) : it(i) {}
        reference operator*() const { return it->second; }
        pointer operator->() const { return &it->second; }
        ValueIterator& operator++() { ++it; return *this; }
        ValueIterator operator++(int) { auto old = *this; ++*this; return old; }
        ValueIterator& operator--() { --it; return *this; }
        ValueIterator operator--(int) { auto old = *this; --*this; return old; }
        bool operator==(const ValueIterator&) const = default;
    };
public:
    using iterator = ValueIterator<typename Tree<K, Item>::iterator>;
    using const_iterator = ValueIterator<typename Tree<K, Item>::const_iterator>;
    iterator begin() { return iterator(data.begin()); }
    iterator end() { return iterator(data.end()); }
    const_iterator begin() const { return const_iterator(data.begin()); }
    const_iterator end() const { return const_iterator(data.end()); }
    auto rbegin() { return std::make_reverse_iterator(end()); }
    auto rend() { return std::make_reverse_iterator(begin()); }
    auto rbegin() const { return std::make_reverse_iterator(end()); }
    auto rend() const { return std::make_reverse_iterator(begin()); }
    bool empty() const { return data.empty(); }
    size_t size() const { return data.size(); }
    Item& front() { return *begin(); }
    const Item& front() const { return *begin(); }
    Item& back() { return *rbegin(); }
    const Item& back() const { return *rbegin(); }
    iterator lower_bound(K key) { return iterator(data.lower_bound(key)); }
    const_iterator lower_bound(K key) const { return const_iterator(data.lower_bound(key)); }
    std::pair<iterator, bool> insert(Item item) {
        auto [it, added] = data.insert({item.*Key, item});
        return {iterator(it), added};
    }
    iterator insert_hint(iterator hint, Item item) {
        return iterator(data.emplace_hint(hint.it, item.*Key, item));
    }
    iterator erase(iterator it) { return iterator(data.erase(it.it)); }
    void pop_back() { data.erase(std::prev(data.end())); }
    void pop_front() { data.erase(data.begin()); }
    void clear() { data.clear(); }
};

// Flat policies preserve the original SlopeDP insertion paths.
template<class Item> struct FlatLeft : std::vector<Item> {
    void add_mass(decltype(Item::key) key, decltype(Item::mass) mass) {
        auto it = this->end();
        while (it != this->begin() && std::prev(it)->key > key) --it;
        if (it != this->begin() && std::prev(it)->key == key) std::prev(it)->mass += mass;
        else this->insert(it, {key, mass});
    }
};
template<class Item> struct FlatRight : std::deque<Item> {
    void add_mass(decltype(Item::key) key, decltype(Item::mass) mass) {
        auto it = std::lower_bound(this->begin(), this->end(), key,
            [](const Item& item, auto k) { return item.key < k; });
        if (it != this->end() && it->key == key) it->mass += mass;
        else this->insert(it, {key, mass});
    }
};
template<class Item, template<class...> class Tree = std::map>
struct TreeMasses : OrderedTree<Item, &Item::key, Tree> {
    void add_mass(decltype(Item::key) key, decltype(Item::mass) mass) {
        auto it = this->lower_bound(key);
        if (it != this->end() && it->key == key) it->mass += mass;
        else this->insert_hint(it, {key, mass});
    }
};
struct FlatSlopeStorage {
    template<class Item> using Left = FlatLeft<Item>;
    template<class Item> using Right = FlatRight<Item>;
};

// Profile stores expose ordered iteration and logarithmic lower_bound.
// Bulk PL operations still rebuild profiles: std::map has no subtree
// aggregates, range maxima or lazy range updates.
template<class Item> struct FlatVertices : std::vector<Item> {
    using std::vector<Item>::operator=;
    auto lower_bound(decltype(Item::k) k) const {
        return std::lower_bound(this->begin(), this->end(), k,
            [](const Item& item, auto key) { return item.k < key; });
    }
    void sort() { std::sort(this->begin(), this->end(), [](auto a, auto b) { return a.k < b.k; }); }
};
template<class Item, template<class...> class Tree = std::map>
struct TreeVertices : OrderedTree<Item, &Item::k, Tree> {
    using Base = OrderedTree<Item, &Item::k, Tree>;
    void reserve(size_t) {}
    void push_back(Item item) {
        auto it = this->lower_bound(item.k);
        if (it != this->end() && it->k == item.k) it->v = std::max(it->v, item.v);
        else this->insert_hint(it, item);
    }
    void insert(typename Base::iterator, Item item) { push_back(item); }
    TreeVertices& operator=(std::vector<Item>&& items) {
        this->clear();
        for (auto item : items) push_back(item);
        return *this;
    }
    void sort() {}  // Already ordered by the tree.
};
struct FlatProfileStorage { template<class Item> using Vertices = FlatVertices<Item>; };
template<template<class...> class Tree = std::map> struct BalancedTreeStorage {
    template<class Item> using Left = TreeMasses<Item, Tree>;
    template<class Item> using Right = TreeMasses<Item, Tree>;
    template<class Item> using Vertices = TreeVertices<Item, Tree>;
};
using MapStorage = BalancedTreeStorage<>;
}  // namespace breakpoint_storage
#endif
