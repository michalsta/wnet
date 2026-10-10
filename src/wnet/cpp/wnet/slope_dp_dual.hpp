#ifndef WNET_SLOPE_DP_DUAL_HPP
#define WNET_SLOPE_DP_DUAL_HPP
#include <algorithm>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>
namespace slopedp_dual {
// Recover a common matching dual from complementary slackness on the chain.
// pi lies in [-tau, 0]. Slack empirical/theoretical anchors force 0/-tau;
// positive/negative net flow fixes the next difference to -gap/+gap.
// Forward feasible intervals and backward selection solve these constraints.
// alpha=pi, beta=-tau-pi then define one common matching dual.
// Saturating arithmetic protects interval propagation at integer limits.
template<class I>
std::vector<I> recover(std::span<const I> e, std::span<const I> t,
                       std::span<const I> u, std::span<const I> y,
                       std::span<const I> s, std::span<const I> gap, I tau) {
    const size_t n = e.size();
    if (t.size() != n || u.size() != n || y.size() != n ||
        s.size() != (n ? n - 1 : 0) || gap.size() != s.size() || tau < 0)
        throw std::invalid_argument("inconsistent recovery inputs");
    auto add = [](I a, I b) {
        if (b > 0 && a > std::numeric_limits<I>::max() - b) return std::numeric_limits<I>::max();
        if (b < 0 && a < std::numeric_limits<I>::min() - b) return std::numeric_limits<I>::min();
        return I(a + b);
    };
    std::vector<I> lower(n, -tau), upper(n, 0), pi(n);
    for (size_t i = 0; i < n; ++i) {
        if (u[i] < 0 || u[i] > e[i] || y[i] < 0 || y[i] > t[i])
            throw std::runtime_error("invalid primal capacity");
        if (u[i] < e[i]) lower[i] = 0;
        if (y[i] < t[i]) upper[i] = -tau;
        if (i) {
            if (gap[i - 1] < 0) throw std::runtime_error("negative chain gap");
            I dl = s[i - 1] < 0 ? gap[i - 1] : -gap[i - 1];
            I dh = s[i - 1] > 0 ? -gap[i - 1] : gap[i - 1];
            lower[i] = std::max(lower[i], add(lower[i - 1], dl));
            upper[i] = std::min(upper[i], add(upper[i - 1], dh));
        }
        if (lower[i] > upper[i]) throw std::runtime_error("no feasible DP dual");
    }
    if (n) pi.back() = upper.back();
    for (size_t i = n; i > 1; --i) {
        size_t j = i - 2;
        I dl = s[j] > 0 ? -gap[j] : (s[j] < 0 ? gap[j] : -gap[j]);
        I dh = s[j] > 0 ? -gap[j] : gap[j];
        I l = std::max(lower[j], add(pi[j + 1], -dh));
        I h = std::min(upper[j], add(pi[j + 1], -dl));
        if (l > h) throw std::runtime_error("inconsistent backward interval");
        pi[j] = h;
    }
    return pi;
}

}
#endif
