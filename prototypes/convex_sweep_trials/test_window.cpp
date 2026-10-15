#include "trials.hpp"
#include <cassert>
#include <iostream>
#include <random>
using namespace sweep_trials;
int main() {
  std::mt19937 random(121);
  for (size_t trial = 0; trial < 10000; ++trial) {
    k::ExactPLF f;
    I K = 0, value = 0;
    f.vs.push_back({0, 0});
    for (int i = 0; i < 8; ++i) {
      const I count = 1 + random() % 8;
      K += count;
      value += I(int(random() % 21) - 10) * count;
      f.vs.push_back({K, value});
    }
    const I q = random() % (K + 20);
    auto result = k::windowed_max(f, q);
    for (I g = 0; g <= K; ++g) {
      I expected = INT64_MIN;
      for (I u = g; u <= std::min(K, g + q); ++u)
        expected = std::max(expected, f.eval(u));
      assert(result.eval(g) == expected);
    }
  }
  // The streaming profit cursor must reproduce the old evaluator exactly,
  // including repeated queries and non-integral interpolation fallbacks.
  for (size_t trial = 0; trial < 10000; ++trial) {
    k::PLF f;
    I K = 0, value = 0;
    f.vs.push_back({0, 0});
    for (int j = 0; j < 8; ++j) {
      K += 1 + random() % 8;
      value += I(int(random() % 101) - 50);
      f.vs.push_back({K, value});
    }
    k::CurveCursor cursor(f);
    for (I q = 0; q <= K; ++q) {
      assert(cursor.eval(q) == f.eval(q));
      assert(cursor.eval(q) == f.eval(q));
    }
  }
  std::vector<k::Event> large{{0, 0, 1, 0},
                              {4503599627370495., 0, 1000000000000000000LL, 1}};
  MomentIndex index(large, 0);
  assert(index.big);
  FastMomentCurve t(index, 1000000000000000001LL, 1000000000000000000LL,
                    4503599627370495LL, 1);
  k::CurveCursor cursor(t);
  for (I q : {0LL, 1LL, 2LL, 100LL, 1000000000LL, 500000000000000000LL,
              999999999999999999LL, 1000000000000000000LL})
    assert(cursor.eval(q) == q);
  std::cout << "10000 exact-window cases, 10000 cursor interpolation cases and "
               "wide-moment cursors passed\n";
}
