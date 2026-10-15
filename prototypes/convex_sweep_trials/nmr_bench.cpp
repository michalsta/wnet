#include "nmr_trials.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
using namespace nmr_trials;
int main(int argc, char **argv) {
    if (argc < 4)
        throw std::invalid_argument(
            "fixture variant solve|dual|gains|both [repetitions] [pair_limit]");
    auto start = std::chrono::steady_clock::now();
    Fixture fixture(argv[1]);
    auto v = variant(argv[2]);
    std::string phase = argv[3];
    int reps = argc > 4 ? std::stoi(argv[4]) : 3;
    size_t limit = argc > 5 ? std::stoull(argv[5]) : 8000000;
    int warmup = argc > 6 ? std::stoi(argv[6]) : 1;
    require(reps > 0 && warmup >= 0 && warmup <= 1, "repetitions and warmup");
    std::cout << std::setprecision(17) << "{\"event\":\"ready\",\"scale\":" << fixture.scale
              << ",\"p\":" << fixture.p << ",\"input_events\":" << fixture.original_events
              << ",\"grouped_events\":" << fixture.grouped_events << ",\"setup_ms\":"
              << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                     .count()
              << "}\n"
              << std::flush;
    std::vector<double> samples;
    I reference = 0;
    uint64_t refhash = 0;
    for (int sample = 0; sample < reps + warmup; ++sample) {
        std::cout << "{\"event\":\"sample_start\",\"sample\":" << sample << "}\n" << std::flush;
        auto begin = std::chrono::steady_clock::now();
        W profit = 0;
        uint64_t hash = 1469598103934665603ULL;
        size_t pairs = 0;
        double trace_ms = 0, original_dual_ms = 0, grouped_dual_ms = 0;
        for (const auto &in : fixture.components) {
            auto r = run(in, v, phase, limit);
            profit += r.profit;
            pairs += r.pairs;
            trace_ms += r.trace_ms;
            original_dual_ms += r.original_dual_ms;
            grouped_dual_ms += r.grouped_dual_ms;
            for (auto g : r.gains) {
                hash ^= uint64_t(g);
                hash *= 1099511628211ULL;
            }
        }
        auto ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin)
                .count();
        I value = static_cast<I>(profit);
        if (sample)
            require((phase == "gains" || reference == value) && refhash == hash, "repeat mismatch");
        if (sample >= warmup)
            samples.push_back(ms);
        reference = value;
        refhash = hash;
        std::cout << "{\"event\":\"sample_done\",\"sample\":" << sample << ",\"ms\":" << ms
                  << ",\"profit\":" << (phase == "gains" ? "null" : std::to_string(value))
                  << ",\"gain_hash\":" << hash << ",\"pairs\":" << pairs
                  << ",\"trace_ms\":" << trace_ms << ",\"original_dual_ms\":" << original_dual_ms
                  << ",\"grouped_dual_ms\":" << grouped_dual_ms << "}\n"
                  << std::flush;
    }
    std::sort(samples.begin(), samples.end());
    std::cout << "{\"event\":\"result\",\"median_ms\":" << samples[samples.size() / 2]
              << ",\"min_ms\":" << samples.front() << ",\"max_ms\":" << samples.back()
              << ",\"profit\":" << (phase == "gains" ? "null" : std::to_string(reference))
              << ",\"gain_hash\":" << refhash << "}\n";
}
