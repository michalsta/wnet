# Alternative balanced-tree storage

Both chain solvers accept an optional storage choice:

```python
from wnet.wnet_cpp import SlopeDP, ConvexSweep

slope = SlopeDP(dual_mode="cluster", storage="tree")
sweep = ConvexSweep(storage="tree")
# Pass either configuration as WassersteinNetwork(..., solver=...).
network.solve(weights, with_duals=True)  # ConvexSweep certificates
cut = network.dual_cut()
```

`storage="flat"` remains the default. SlopeDP's `lazy` and `cluster` dual modes
work with either choice. ConvexSweep's cost-only and traced specializations,
marginal calculations, position updates and precision rebuilds all use the
selected storage. Trash semantics, integer arithmetic and quantization are
shared. Tree selection is independent of whether duals are requested.

The tree variant uses ordered breakpoint maps throughout the kernels. Sorted
vectors remain useful as temporary output buffers and candidate lists; the
live DP profiles and traced profiles use the selected store. There is no
conversion to the flat solver to evaluate a tree profile.

## Replacing the tree in C++

[`ordered_breakpoints.hpp`](../src/wnet/cpp/wnet/ordered_breakpoints.hpp)
defines `BalancedTreeStorage<Tree>`, defaulting to `std::map`. Its
`OrderedTree<Item, Key, Tree>` adapter exposes payload iterators, ordered
lookup, insertion and erasure. The kernels template on storage policies;
there is no virtual dispatch or per-breakpoint runtime storage choice.

```cpp
using Storage = breakpoint_storage::BalancedTreeStorage<MyMap>;
slopedp_detail::ConvexPL<int64_t, Storage> slope;
using Profile = convex_sweep::BasicPLF<Storage>;
auto result = convex_sweep::sweep_solve<true, Profile>(events, profit, radius);
```

`MyMap` must be a map-like class template instantiated as `MyMap<Key, Item>`.
It must provide `iterator`, `const_iterator`, bidirectional ordered iteration,
`begin/end`, `empty/size/clear`, `lower_bound`, `insert`, `emplace_hint`, and
`erase(iterator)` with ordinary map semantics. Iterators expose `first/second`.
A different API can be accommodated inside the adapter. Breakpoint keys
stay immutable; SlopeDP's bulk coordinate shifts use its existing offsets.
Custom policies are selected at C++ compile time; the Python `tree` choice
currently selects `std::map`.

[`test_tree_breakpoints.cpp`](../prototypes/test_tree_breakpoints.cpp) exercises
an instrumented second policy with both kernels and compares 40,000 SlopeDP
transitions against flat storage. The ConvexSweep native oracle checks flat
and tree costs, traceback and adjoint marginals on 3,000 randomized cases.
Python tests also compare certificates against independent continuous LPs.

## Costs and limitations

For `B` live breakpoints, map lookup and insertion cost `O(log B)`, with
`O(B)` storage and larger per-node constants than flat containers. SlopeDP
replaces potentially linear middle insertions with logarithmic tree operations;
its existing end removals, offset shifts and final evaluation remain shared.
This does not imply that a map is faster on typical inputs with small profiles.

ConvexSweep still scans and rebuilds general piecewise-linear profiles.
Map reconstruction costs `O(B log B)`; storing breakpoints in a balanced tree
alone does not turn the sweep into an `O(N log N)` algorithm. Traced solves
retain a sum of profile sizes across events, potentially quadratic, and free
those snapshots after native dual recovery. Potential recovery is unchanged.

Further speed improvements need an augmented tree and richer operations such
as range maxima, lazy transformations or rank/mass queries. A plain map cannot
provide those efficiently through its standard interface. Keeping the adapter
small gives a correctness baseline for tree swaps; adding aggregates would
also require updating the relevant profile operations.

[`bench_tree_storage.cpp`](../prototypes/bench_tree_storage.cpp) compares
SlopeDP and ConvexSweep kernel times and validates matching objectives. It
accepts a position count and profitable radius, for example `1000 8` and
`1000 1000` for narrow and wide windows. It excludes graph construction and
potential recovery, so it is not an end-to-end optimizer benchmark.
