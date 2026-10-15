#ifndef SWEEP_TRIAL_MOMENTS_HPP
#define SWEEP_TRIAL_MOMENTS_HPP
#include "stats.hpp"
#include <algorithm>
#include <array>
#include <boost/multiprecision/cpp_int.hpp>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>
namespace sweep_trials {
using I = int64_t;
using Big = boost::multiprecision::int256_t;
#if defined(__SIZEOF_INT128__)
using Small = __int128;
#else
using Small = boost::multiprecision::int128_t;
#endif
// Prefix sums are centered on the first coordinate. Use 256-bit moments
// when cancellation terms cannot safely fit 128 bits, even if final costs do.
template <class M> struct MomentSums {
  std::vector<M> x{0}, xx{0};
  template <class Events> MomentSums(const Events &events, int side, I origin) {
    for (const auto &e : events)
      if (e.side == side && e.count > 0) {
        const M d = M(I(e.pos)) - origin;
        x.push_back(x.back() + d * e.count);
        xx.push_back(xx.back() + d * d * e.count);
      }
  }
};
struct MomentIndex {
  std::vector<I> mass{0}, pos;
  std::unique_ptr<MomentSums<Small>> small;
  std::unique_ptr<MomentSums<Big>> big;
  I origin = 0;
  template <class Events> MomentIndex(const Events &events, int side) {
    if (!events.empty())
      origin = I(events.front().pos);
    Big extent = 0;
    for (const auto &e : events) {
      extent = std::max(extent, Big(I(e.pos)) - origin);
      if (e.side == side && e.count > 0) {
        mass.push_back(mass.back() + e.count);
        pos.push_back(I(e.pos));
      }
    }
    if (extent * extent * mass.back() * 16 < (Big(1) << 126))
      small = std::make_unique<MomentSums<Small>>(events, side, origin);
    else
      big = std::make_unique<MomentSums<Big>>(events, side, origin);
  }
  template <class M> M prefix(I rank, I z, const MomentSums<M> &sums) const {
    const size_t j = size_t(std::upper_bound(mass.begin(), mass.end(), rank) -
                            mass.begin() - 1);
    const I partial = rank - mass[j];
    M x = sums.x[j], xx = sums.xx[j];
    if (partial) {
      const M d = M(pos[j]) - origin;
      x += d * partial;
      xx += d * d * partial;
    }
    const M d = M(z) - origin;
    return xx - 2 * d * x + d * d * rank;
  }
  I benefit(I first, I last, I z, I tau) const {
    auto calculate = [&](const auto &sums) -> I {
      using M = typename std::remove_cvref_t<decltype(sums.x)>::value_type;
      const M result = M(tau) * (last - first) - prefix(last, z, sums) +
                       prefix(first, z, sums);
      if (result < M(std::numeric_limits<I>::min()) ||
          M(std::numeric_limits<I>::max()) < result)
        throw std::overflow_error("moment profit exceeds int64");
      return static_cast<I>(result);
    };
    return small ? calculate(*small) : calculate(*big);
  }
};
struct MomentBank {
  MomentIndex sides[2];
  I end[2] = {0, 0};
  int own = 0;
  template <class Events>
  explicit MomentBank(const Events &events)
      : sides{MomentIndex(events, 0), MomentIndex(events, 1)} {}
};
struct MomentCurve {
  const MomentIndex *index;
  I end, domain, z, tau;
  size_t first, last;
  MomentCurve(const MomentIndex &idx, I end_, I domain_, I z_, I tau_)
      : index(&idx), end(end_), domain(domain_), z(z_), tau(tau_) {
    first = size_t(
        std::lower_bound(idx.mass.begin(), idx.mass.end(), end - domain) -
        idx.mass.begin());
    last = size_t(std::lower_bound(idx.mass.begin(), idx.mass.end(), end) -
                  idx.mass.begin());
    if (first > last || last >= idx.mass.size() ||
        idx.mass[first] != end - domain || idx.mass[last] != end)
      throw std::logic_error("moment suffix is not aligned to pending blocks");
  }
  I K() const { return domain; }
  size_t knot_count() const { return last - first + 1; }
  I knot(size_t k) const { return end - index->mass[last - k]; }
  I eval(I k) const {
    TRIAL_COUNT(moment_queries, 1);
    k = std::clamp<I>(k, 0, domain);
    return index->benefit(end - k, end, z, tau);
  }
};

// Second moment prototype: cache the fixed end prefix and evaluate knots
// directly. No binary search is needed when the caller already has a knot.
struct FastMomentCurve : MomentCurve {
  Small small_end = 0;
  Big big_end = 0;
  template <class M> M at(size_t j, const MomentSums<M> &sums) const {
    const M d = M(z) - index->origin;
    return M(tau) * index->mass[j] - sums.xx[j] + 2 * d * sums.x[j] -
           d * d * index->mass[j];
  }
  FastMomentCurve(const MomentIndex &idx, I end, I domain, I z, I tau)
      : MomentCurve(idx, end, domain, z, tau) {
    if (index->small)
      small_end = at(last, *index->small);
    else
      big_end = at(last, *index->big);
  }
  template <class M> I narrow(M v) const {
    if (v < M(std::numeric_limits<I>::min()) ||
        M(std::numeric_limits<I>::max()) < v)
      throw std::overflow_error("fast moment profit exceeds int64");
    return static_cast<I>(v);
  }
  I knot_value(size_t i) const {
    if (index->small)
      return narrow(small_end - at(last - i, *index->small));
    return narrow(big_end - at(last - i, *index->big));
  }
  I eval(I k) const {
    TRIAL_COUNT(moment_queries, 1);
    if (k <= 0)
      return 0;
    k = std::min(k, domain);
    const I rank = end - k;
    const size_t j =
        size_t(std::upper_bound(index->mass.begin() + first,
                                index->mass.begin() + last + 1, rank) -
               index->mass.begin() - 1);
    const I partial = rank - index->mass[j];
    auto prefix = [&](const auto &sums, const auto &ending) -> I {
      using M = std::remove_cvref_t<decltype(ending)>;
      M value = at(j, sums);
      if (partial) {
        const M d = M(z) - index->pos[j];
        value += (M(tau) - d * d) * partial;
      }
      return narrow(M(ending - value));
    };
    if (index->small)
      return prefix(*index->small, small_end);
    return prefix(*index->big, big_end);
  }
};

} // namespace sweep_trials
#endif
