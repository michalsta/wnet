# Solver implementation TODOs

- The alternative `storage="tree"` implementations of SlopeDP and ConvexSweep
  now use a shared, templated ordered-breakpoint interface with `std::map` as
  the first tree policy. Keep flat storage as the default. See
  [Balanced-tree storage](balanced_tree_storage.md).
- Explore augmented trees with subtree masses, range maxima and lazy range
  updates. The map baseline does not remove ConvexSweep's full-profile scans
  or snapshots. Establish bounds on breakpoint growth and benchmark graph
  construction, transport solves, certificate recovery, gradients, and peak
  memory on sparse and wide-window inputs before selecting a new default.
