// Optional full-fixture verification of flow sensitivities by fresh +1 solves.
// Timings from this diagnostic are not backend benchmark measurements.
#include "nmr_trials.hpp"
#include <iostream>
using namespace nmr_trials;
int main(int argc, char **argv) {
    if (argc < 2)
        throw std::invalid_argument("fixture [probes_per_component=4]");
    Fixture fixture(argv[1]);
    size_t budget = argc > 2 ? std::stoul(argv[2]) : 4, checked = 0;
    W total = 0;
    for (size_t component = 0; component < fixture.components.size(); ++component) {
        auto in = fixture.components[component];
        auto full = flow(in, true, true, 8000000);
        total += W(full.profit);
        std::vector<size_t> candidates;
        for (size_t i = 0; i < in.pos.size(); ++i)
            if (in.theo[i] || in.t[i])
                candidates.push_back(i);
        if (candidates.empty())
            continue;
        std::vector<size_t> probes;
        auto add = [&](size_t i) {
            for (auto j : probes)
                if (in.pos[j] == in.pos[i])
                    return;
            probes.push_back(i);
        };
        add(*std::min_element(candidates.begin(), candidates.end(), [&](auto a, auto b) {
            I x = full.gains[a], y = full.gains[b];
            return std::pair(x ? 0 : 1, x) < std::pair(y ? 0 : 1, y);
        }));
        add(*std::max_element(candidates.begin(), candidates.end(),
                              [&](auto a, auto b) { return full.gains[a] < full.gains[b]; }));
        auto zero =
            std::find_if(candidates.begin(), candidates.end(), [&](auto i) { return !in.t[i]; });
        if (zero != candidates.end())
            add(*zero);
        add(*std::max_element(candidates.begin(), candidates.end(),
                              [&](auto a, auto b) { return in.t[a] < in.t[b]; }));
        auto gainzero = std::find_if(candidates.begin(), candidates.end(),
                                     [&](auto i) { return !full.gains[i]; });
        if (gainzero != candidates.end())
            add(*gainzero);
        for (auto i : candidates) {
            if (probes.size() >= budget)
                break;
            add(i);
        }
        for (size_t k = 0; k < std::min(budget, probes.size()); ++k) {
            auto i = probes[k];
            ++in.t[i];
            auto more = flow(in, false, false, 8000000);
            --in.t[i];
            I gain = more.profit - full.profit;
            require(gain == full.gains[i], "full flow +1 gain mismatch");
            ++checked;
            std::cout << "component=" << component << " index=" << i << " gain=" << gain
                      << " verified\n"
                      << std::flush;
        }
    }
    std::cout << "profit=" << static_cast<I>(total) << " verified_probes=" << checked << "\n";
}
