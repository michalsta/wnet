# Augmented-tree storage

```python
from wnet.wnet_cpp import SlopeDP, ConvexSweep

slope = SlopeDP(storage="augmented")
sweep = ConvexSweep(storage="augmented")
```

These are alternatives to the flat and map kernels. They share the graph,
integer cost model, flow reconstruction and certificate recovery with those
kernels. SlopeDP's lazy and cluster dual modes both work. ConvexSweep retains
its separate compile-time cost-only and traced paths and eager dual recovery.
The flat policy remains the default: lower asymptotic costs for individual
operations do not guarantee faster solves, and these trees allocate larger
nodes and perform reference counting.

## SlopeDP

[`augmented_slope_dp.hpp`](../src/wnet/cpp/wnet/augmented_slope_dp.hpp)
stores the left/right slope masses in ordered treaps. Each node caches total
mass and its first moment, using the existing wide integer arithmetic. A
weighted split can divide the mass at one breakpoint. `addAbs` transfers a
whole chunk between trees using key splits, cumulative-mass splits, a lazy
coordinate shift, and ordered joins. It updates the minimum value from the
chunk's mass and moment, without visiting each removed breakpoint.

For `B` live breakpoints, insertion, key/mass splitting, joining and point
queries take expected `O(log B)` structural work. There are only a few such
operations per DP transition. Endpoint, total-mass and total-moment queries
are `O(1)`; coordinate shifts are lazy. Removing a uniquely owned discarded
subtree also costs `O(D)` to reclaim its `D` nodes. Such reclamation is separate
from the structural split cost. Treaps provide expected bounds, rather than
the worst-case guarantees of AVL or red-black trees.

## ConvexSweep

[`augmented_profiles.hpp`](../src/wnet/cpp/wnet/augmented_profiles.hpp)
uses an ordered vertex treap; adjacent vertices define linear segments. Nodes
cache the maximum value and the first/last maximizing coordinates. Coordinate
shifts, reflection and profile value shifts have lazy tags. This representation
uses vertex coordinates rather than an implicit rope of segment lengths.

Evaluation and interval argmax take expected `O(log B)`. Truncation and left
clamping split the tree and interpolate one boundary value. Shift/append and
flat extension insert boundary vertices. These operations use expected
`O(log B)` structural work plus local collinearity cleanup; deleting `D`
collinear vertices costs `O(D log B)` and reclaiming discarded nodes costs
`O(D)`. Reflection and copying a finalized profile take `O(1)` root work.

Roots share unchanged subtrees; edits copy only affected paths. Full traversal
materializes a vector once per immutable profile and shares that cache with
copies. New profiles accumulate sorted vertices in a vector and build a treap
in `O(B)` using a Cartesian-tree construction.

General pointwise maxima, suffix maxima, addition/subtraction and the main
sliding-window transform still scan and rebuild profiles. Pointwise maxima
merge the two sorted vertex streams, evaluate known segments, find integer
crossings, and build the result in linear work for fixed-width coordinates
(up to 64 bisection steps per crossing). The sliding-window transform's rare
exact-value fallback uses interval maxima instead of rescanning all vertices.
These changes do not establish an `O(N log N)` bound for the complete sweep.
Traceback retains shared roots and traversal caches, but rebuilt profiles can
still require a sum of profile sizes across events, potentially quadratic.
Persistent node memory can exceed flat snapshots considerably.

## Replacing the augmented tree

The shared policy in
[`augmented_treap.hpp`](../src/wnet/cpp/wnet/augmented_treap.hpp) is independent
of the two solver adapters:

```cpp
using Storage = breakpoint_storage::AugmentedStorage<MyAugmentedTree>;
using Slope = Storage::Slope<int64_t>;
using Profile = convex_sweep::BasicPLF<Storage>;
auto result = convex_sweep::sweep_solve<true, Profile>(events, profit, radius);
```

`MyAugmentedTree<bool Mass>` must provide the same semantic interface as
`AugmentedTreap`: persistent copies; `Entry {key,value}` and `Totals
{mass,moment}`; ordered endpoint/size queries; sorted construction/traversal;
insertion/erasure; consuming key splits and ordered joins; lazy shifts and
reflection; predecessor/successor queries. The mass specialization additionally
provides total mass, wide first moment, prefix aggregates and consuming
cumulative-mass splits. The vertex specialization provides value shifts and
interval vertex maxima with the earliest maximizing coordinate. Equal-key
joins sum masses or take the maximum vertex value. A plain `std::map` cannot
implement these augmented operations efficiently through its standard API;
it remains available through the separate `BalancedTreeStorage` policy.

The native primitive tests cover persistence, mass splits, heap order, moments,
lazy transforms and interval maxima. Solver tests compare flat/map/augmented
costs, traceback, certificates and marginals against independent oracles,
including the software wide-integer implementation with UBSan.
