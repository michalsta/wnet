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
  std::cout << "10000 exact generic-window cases passed\n";
}
