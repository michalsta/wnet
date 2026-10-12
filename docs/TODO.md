# Solver implementation TODOs

- Alternative `storage="tree"` and `storage="augmented"` implementations of
  SlopeDP and ConvexSweep are implemented with compile-time storage policies.
  Keep flat storage as the default. See [Balanced-tree storage](balanced_tree_storage.md)
  and [Augmented-tree storage](augmented_tree_storage.md).
- Reduce allocation and reference-count overhead in augmented trees, for example
  with arena-backed nodes or a cache-conscious tree. Preserve persistent traced
  profiles and wide integer first moments when replacing the policy.
- Establish bounds on breakpoint growth. ConvexSweep's general full-profile
  transforms still scan/rebuild profiles; persistence does not eliminate all
  retained profile data. Benchmark representative deconvolution inputs before
  selecting a new default.
