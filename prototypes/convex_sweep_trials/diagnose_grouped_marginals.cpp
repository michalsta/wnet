// Intentionally bypass the safe dispatcher to reproduce merged-adjoint loss.
#include "fixture.hpp"
#include <iostream>
using namespace sweep_trials;
int main(int argc, char **argv) {
  Fixture f(argv[1]);
  size_t mismatches = 0;
  for (size_t c = 0; c < f.components.size(); ++c) {
    auto in = f.components[c];
    auto a = gains(in, variant("baseline")),
         b = gains_impl<k::PLF, k::ReferencePolicy>(in, variant("group"));
    I base = solve(in, variant("baseline")).profit;
    for (size_t i = 0; i < a.size(); ++i)
      if (a[i] != b[i]) {
        ++mismatches;
        std::cout << "component=" << c << " i=" << i << " q=" << in.pos[i]
                  << " t=" << in.t[i] << " old=" << a[i] << " group=" << b[i];
        ++in.t[i];
        auto x = solve(in, variant("baseline")),
             y = solve(in, variant("group"));
        --in.t[i];
        std::cout << " probe=" << x.profit - base
                  << " group-probe=" << y.profit - base << "\n";
        if (mismatches >= 30)
          return 0;
      }
  }
  std::cout << "mismatches=" << mismatches << '\n';
}
