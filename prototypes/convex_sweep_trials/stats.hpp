#ifndef SWEEP_TRIAL_STATS_HPP
#define SWEEP_TRIAL_STATS_HPP
#include <algorithm>
#include <cstdint>
namespace sweep_trials {
struct Stats {
  uint64_t normalized_vertices = 0, normalizations = 0, pointwise_merges = 0,
           window_passes = 0, profit_knots = 0, moment_queries = 0,
           largest_profile = 0;
};
inline thread_local Stats stats;
} // namespace sweep_trials
#ifdef SWEEP_TRIAL_STATS
#define TRIAL_COUNT(field, n) (sweep_trials::stats.field += (n))
#define TRIAL_MAX(field, n)                                                    \
  (sweep_trials::stats.field =                                                 \
       std::max<uint64_t>(sweep_trials::stats.field, (n)))
#else
#define TRIAL_COUNT(field, n) ((void)0)
#define TRIAL_MAX(field, n) ((void)0)
#endif
#endif
