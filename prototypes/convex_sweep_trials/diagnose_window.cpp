// Diagnostic of an existing generic-window limitation, separate from the
// tested reachable sweep states. Deliberately uses production code directly.
#include "wnet/convex_sweep.hpp"
#include <iostream>
int main() {
  convex_sweep::PLF f;
  f.vs = std::vector<convex_sweep::PLF::V>{
      {0, 200}, {2, 182}, {4, 184}, {12, 112}};
  const int64_t lower = 1, upper = 11;
  auto got = convex_sweep::windowed_max(f, 10).eval(lower);
  int64_t expected = INT64_MIN;
  for (int64_t k = lower; k <= upper; ++k)
    expected = std::max(expected, f.eval(k));
  std::cout << "production generic window: result=" << got
            << " brute_max=" << expected << " mismatch=" << (got != expected)
            << '\n';
  // This is a diagnostic, not an assertion that the discrepancy is correct.
  return 0;
}
