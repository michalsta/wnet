#include "fixture.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
using namespace sweep_trials;
int main(int argc, char **argv) {
  if (argc < 4)
    throw std::invalid_argument(
        "bench fixture variant solve|dual|gains [repetitions]");
  auto started = std::chrono::steady_clock::now();
  Fixture fixture(argv[1]);
  const auto v = variant(argv[2]);
  const std::string phase = argv[3];
  const int repeats = argc > 4 ? std::stoi(argv[4]) : 3;
  if (repeats < 1)
    throw std::invalid_argument("at least one measured repetition is required");
  if (v.moments && fixture.p != 2)
    throw std::invalid_argument("moments requires quadratic model");
  std::cout << std::setprecision(17)
            << "{\"event\":\"ready\",\"scale\":" << fixture.scale
            << ",\"p\":" << fixture.p
            << ",\"events\":" << fixture.original_events
            << ",\"grouped_events\":" << fixture.grouped_events
            << ",\"components\":" << fixture.components.size()
            << ",\"setup_ms\":"
            << std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - started)
                   .count()
            << "}\n"
            << std::flush;
  std::vector<double> samples;
  I reference = 0;
  uint64_t reference_hash = 0;
  for (int sample = 0; sample <= repeats; ++sample) {
    std::cout << "{\"event\":\"sample_start\",\"sample\":" << sample << "}\n"
              << std::flush;
    sweep_trials::stats = {};
    const auto begin = std::chrono::steady_clock::now();
    W total = 0;
    uint64_t hash = 1469598103934665603ULL;
    size_t trace_vertices = 0;
    for (const auto &in : fixture.components) {
      if (phase == "solve") {
        auto r = solve(in, v);
        total += r.profit;
      } else if (phase == "dual") {
        auto r = solve<true>(in, v);
        dual(in, r);
        total += r.profit;
        trace_vertices += r.trace_vertices;
      } else if (phase == "gains") {
        for (I g : gains(in, v)) {
          hash ^= static_cast<uint64_t>(g);
          hash *= 1099511628211ULL;
        }
      } else
        throw std::invalid_argument("unknown benchmark phase");
    }
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - begin)
                          .count();
    const I result = static_cast<I>(total);
    if (sample && (reference != result || reference_hash != hash))
      throw std::runtime_error("repeated result mismatch");
    reference = result;
    reference_hash = hash;
    if (sample)
      samples.push_back(ms);
    std::cout << "{\"event\":\"sample_done\",\"sample\":" << sample
              << ",\"ms\":" << ms << ",\"profit\":"
              << (phase == "gains" ? "null" : std::to_string(result))
              << ",\"gain_hash\":" << hash
              << ",\"trace_vertices\":" << trace_vertices
#ifdef SWEEP_TRIAL_STATS
              << ",\"normalized_vertices\":" << stats.normalized_vertices
              << ",\"normalizations\":" << stats.normalizations
              << ",\"profit_knots\":" << stats.profit_knots
              << ",\"moment_queries\":" << stats.moment_queries
              << ",\"largest_profile\":" << stats.largest_profile
              << ",\"largest_state\":" << stats.largest_state
              << ",\"largest_pending_blocks\":" << stats.largest_pending_blocks
#endif
              << "}\n"
              << std::flush;
  }
  std::sort(samples.begin(), samples.end());
  std::cout << "{\"event\":\"result\",\"median_ms\":"
            << samples[samples.size() / 2] << ",\"min_ms\":" << samples.front()
            << ",\"max_ms\":" << samples.back() << ",\"profit\":"
            << (phase == "gains" ? "null" : std::to_string(reference))
            << ",\"gain_hash\":" << reference_hash << ",\"real_cost\":";
  if (phase == "gains")
    std::cout << "null";
  else
    std::cout << (fixture.base - reference) /
                     (double(fixture.scale) * fixture.intensity);
  std::cout << "}\n";
}
