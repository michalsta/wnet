# Solver implementation TODOs

- Provide alternative balanced-tree implementations of both SlopeDP and
  ConvexSweep. Keep the current implementations available for comparison.
  Preserve integer arithmetic, quantization, trash semantics, and dual-cut
  validity. Establish bounds on breakpoint growth and benchmark construction,
  transport solves, certificate recovery, gradients, and peak memory on sparse
  and wide-window inputs before selecting a default.
