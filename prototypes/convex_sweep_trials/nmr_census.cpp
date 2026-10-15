#include "nmr_common.hpp"
#include <iostream>
using namespace nmr_trials;
int main(int argc, char **argv) {
    Fixture f(argv[1]);
    size_t a = 0, b = 0;
    for (const auto &in : f.components) {
        Coordinates c(in);
        for (bool probes : {false, true}) {
            std::vector<double> t;
            for (size_t j = 0; j < c.pos.size(); ++j)
                if (c.t[j] || (probes && c.theo[j]))
                    t.push_back(c.pos[j]);
            size_t lo = 0, hi = 0, count = 0;
            double r = in.model.radius(in.tau);
            for (size_t i = 0; i < c.pos.size(); ++i)
                if (c.e[i]) {
                    while (lo < t.size() && t[lo] <= c.pos[i] - r)
                        ++lo;
                    hi = std::max(hi, lo);
                    while (hi < t.size() && t[hi] < c.pos[i] + r)
                        ++hi;
                    count += hi - lo;
                }
            if (probes)
                b += count;
            else
                a += count;
        }
    }
    std::cout << "{\"pairs_solve\":" << a << ",\"pairs_probe\":" << b << ",\"scale\":" << f.scale
              << ",\"p\":" << f.p << ",\"input_events\":" << f.original_events
              << ",\"grouped_events\":" << f.grouped_events << "}\n";
}
