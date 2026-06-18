#ifndef WNET_SCALING_HPP
#define WNET_SCALING_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "distances.hpp"
#include "distribution.hpp"

// Scaler — single source of truth for the distance/intensity scale factors that
// turn real-valued spectra into the scaled-integer min-cost-flow network.
//
// Defaults reproduce wnetdeconv's precision-driven behaviour (two independent
// factors derived from a relative `precision` target, an int64-overflow cap,
// a distance-resolution guard and a per-spectrum intensity-loss guard).  The
// `tie_factors` / `max_dropped_fraction` / `enforce_distance_resolution` knobs
// reproduce wnetalign's cruder single-overflow-cap behaviour.
//
// It is a pure advisor: it computes the factors at construction and exposes
// them; the caller still applies them (scale positions, set_intensity_scale,
// unscale total_cost).  Construction is the entire failure surface — guards
// throw std::invalid_argument (→ Python ValueError via nanobind).
//
// The full distributions and the DistanceMetric are taken by the constructor so
// future versions can make position-/metric-aware decisions without an API
// change; today only intensity sums and per-peak intensities are consulted.
template<size_t DIM>
class Scaler {
public:
    using VecDist = VectorDistribution<DIM, double, double>;

    Scaler(
        const VecDist& empirical,
        const std::vector<const VecDist*>& theoretical,
        DistanceMetric metric,
        double max_distance,
        const std::vector<double>& trash_costs,
        double precision = 1e-3,
        double explicit_scale_factor = 0.0,
        bool tie_factors = false,
        double max_dropped_fraction = 0.05,
        bool enforce_distance_resolution = true,
        double max_int = static_cast<double>(int64_t(1) << 60),
        double p = 1.0,
        bool fine_grid_intensity = false
    ) : metric_(metric) {
        const double empirical_sum = empirical.sum_intensities();
        double theoretical_sum = 0.0;
        for (const auto* t : theoretical) theoretical_sum += t->sum_intensities();
        const double max_sum = std::max(empirical_sum, theoretical_sum);
        // The fine-grid policy degrades gracefully to scale 1 for zero total
        // intensity; the other policies divide by max_sum and must reject it.
        if (max_sum <= 0.0 && !fine_grid_intensity)
            throw std::invalid_argument(
                "Scaler: max intensity sum must be positive (are all spectra empty?).");

        // Cost-per-unit-flow bounds: max_distance plus any trash costs.  Trash is
        // optional (the pure-distance WassersteinDistance path supplies none).
        double max_cost = max_distance;
        double min_cost = max_distance;
        for (double c : trash_costs) {
            max_cost = std::max(max_cost, c);
            min_cost = std::min(min_cost, c);
        }
        if (max_cost <= 0.0 || min_cost <= 0.0)
            throw std::invalid_argument(
                "Scaler: cost per unit flow (max_distance / trash costs) must be positive.");

        const bool auto_mode = (explicit_scale_factor <= 0.0) && !tie_factors
                               && !fine_grid_intensity;

        if (explicit_scale_factor > 0.0) {
            // Back-compat: explicit scale_factor sets both factors equal.
            sf_distance_ = explicit_scale_factor;
            sf_intensity_ = explicit_scale_factor;
        } else if (tie_factors) {
            // wnetalign mode: a single overflow-cap factor for both, ignoring
            // precision.  sf = sqrt(max_int / (max_sum * max_cost)).
            const double product = max_sum * max_cost;
            if (std::isinf(product))
                throw std::overflow_error("Scaler: max_sum * max_cost overflows double.");
            const double f = std::sqrt(max_int / product);
            sf_distance_ = f;
            sf_intensity_ = f;
        } else if (fine_grid_intensity) {
            // WassersteinNetwork / WassersteinDistance mode: positions are used
            // as real ground distances (no position pre-scale → sf_distance == 1),
            // only intensities are quantized, onto a fine grid.  Port of the
            // former _auto_intensity_scale().
            sf_distance_ = 1.0;
            sf_intensity_ = _fine_grid_intensity_scale(
                empirical, theoretical, p, max_distance, max_int);
        } else {
            // wnetdeconv (solver) mode: independent precision-driven factors,
            // capped.  Used with p == 1 (positions are pre-scaled by sf_distance).
            //   sf_distance  keeps the relative cost error per arc <= precision
            //                within the smallest cost class.
            //   sf_intensity gives int(total_intensity) = 1/precision flow levels.
            sf_distance_ = 1.0 / (precision * min_cost);
            sf_intensity_ = 1.0 / (precision * max_sum);
            // int64 cap: per-arc int cost <= (sf_distance*sf_intensity) *
            // (max_cost*max_sum); shrink both (keeping their ratio) to fit max_int.
            const double cap_product = max_int / (max_cost * max_sum);
            const double product = sf_distance_ * sf_intensity_;
            if (product > cap_product) {
                const double shrink = std::sqrt(cap_product / product);
                sf_distance_ *= shrink;
                sf_intensity_ *= shrink;
            }
        }

        if (!(sf_distance_ > 0.0 && sf_intensity_ > 0.0))
            throw std::invalid_argument(
                "Scaler: could not compute positive scale factors; check your "
                "data, trash costs, or pass an explicit scale_factor.");

        // Distance-resolution guard (precision auto mode, p == 1 only): the
        // smallest cost-per-unit-flow must survive integer quantization.
        if (enforce_distance_resolution && auto_mode && p == 1.0
            && static_cast<int64_t>(min_cost * sf_distance_) < 1) {
            throw std::invalid_argument(
                "Scaler: auto-computed sf_distance=" + std::to_string(sf_distance_) +
                " cannot represent min cost-per-unit-flow=" + std::to_string(min_cost) +
                " as a positive integer (the graph would have no edges).  "
                "Pass a larger scale_factor or relax precision.");
        }

        // Per-spectrum intensity-loss guard: intensities are quantized to
        // round-toward-zero(intensity * sf_intensity); peaks below one integer
        // unit vanish.  Refuse if any spectrum loses more than the limit.
        // Skipped for the fine-grid policy (warn-not-throw contract).
        // max_dropped_fraction >= 1.0 disables it.
        if (max_dropped_fraction < 1.0 && !fine_grid_intensity) {
            double worst_frac = 0.0;
            std::string worst_name;
            auto check = [&](const VecDist& d, const std::string& name) {
                const double total = d.sum_intensities();
                if (total <= 0.0) return;
                double kept = 0.0;
                for (double v : d.get_intensities())
                    kept += std::trunc(v * sf_intensity_);
                kept /= sf_intensity_;
                const double frac = (total - kept) / total;
                if (frac > worst_frac) { worst_frac = frac; worst_name = name; }
            };
            check(empirical, "empirical_spectrum");
            for (size_t i = 0; i < theoretical.size(); ++i)
                check(*theoretical[i], "theoretical_spectra[" + std::to_string(i) + "]");
            if (worst_frac > max_dropped_fraction)
                throw std::invalid_argument(
                    "Integer intensity quantization at sf_intensity=" +
                    std::to_string(sf_intensity_) + " would drop " +
                    std::to_string(worst_frac * 100.0) + "% of " + worst_name +
                    "'s total intensity (limit " +
                    std::to_string(max_dropped_fraction * 100.0) + "%): peaks below "
                    "one integer unit floor to zero supply, leaving the transport "
                    "network nearly empty.  Pass a larger scale_factor, relax "
                    "precision, or allow the loss.");
        }
    }

    double sf_distance() const { return sf_distance_; }
    double sf_intensity() const { return sf_intensity_; }
    // Geometric mean — matches the legacy `scale_factor` alias in both packages
    // (quadratic unscaling sf_distance*sf_intensity == scale_factor**2).
    double scale_factor() const { return std::sqrt(sf_distance_ * sf_intensity_); }
    double ftol() const { return 1.0 / (sf_distance_ * sf_intensity_); }
    DistanceMetric metric() const { return metric_; }

private:
    // A fine but modest target for the total scaled flow: large enough to
    // resolve fractional supplies, small enough to leave int64 headroom.
    static constexpr double FINE_GRID_TARGET_FLOW =
        static_cast<double>(int64_t(1) << 30);

    // Fine-grid intensity scale for the WassersteinNetwork / WassersteinDistance
    // path: map real intensities onto a fine integer supply grid without
    // overflowing the int64 cost accumulator.  Returns 1.0 (bit-compatible with
    // the integer backend) when intensities are already integer-valued.
    //
    // The cap leaves room for the network's own cost scale: it bounds the
    // intensity scale by max_int / (max_real_cost * total), where max_real_cost
    // = cost_bound^p.  pick_cost_scale() then chooses the cost scale against the
    // same accumulator with total_flow already including this intensity scale,
    // so cost scale stays >= 1 and the accumulator never overflows — this is
    // what lets p != 1 finely scale intensities too (no rounding-to-1 pin).
    static double _fine_grid_intensity_scale(
        const VecDist& empirical,
        const std::vector<const VecDist*>& theoretical,
        double p, double max_distance, double max_int) {
        auto all_int = [](const VecDist& d) {
            for (double v : d.get_intensities())
                if (v != std::round(v)) return false;
            return true;
        };
        bool all_integer = all_int(empirical);
        for (const auto* t : theoretical) all_integer = all_integer && all_int(*t);
        if (all_integer) return 1.0;

        double total = 0.0;
        auto add_total = [&](const VecDist& d) {
            for (double v : d.get_intensities()) total += std::abs(v);
        };
        add_total(empirical);
        for (const auto* t : theoretical) add_total(*t);
        if (!(total > 0.0)) return 1.0;

        // span = sum over dims of (max - min) position across all distributions:
        // a conservative bound on the ground distance any unit of flow travels.
        std::array<double, DIM> gmin, gmax;
        gmin.fill(std::numeric_limits<double>::infinity());
        gmax.fill(-std::numeric_limits<double>::infinity());
        auto box = [&](const VecDist& d) {
            for (const auto& pt : d.get_positions())
                for (size_t k = 0; k < DIM; ++k) {
                    gmin[k] = std::min(gmin[k], pt[k]);
                    gmax[k] = std::max(gmax[k], pt[k]);
                }
        };
        box(empirical);
        for (const auto* t : theoretical) box(*t);
        double span = 0.0;
        for (size_t k = 0; k < DIM; ++k) span += (gmax[k] - gmin[k]);

        const double cost_bound = std::max(std::min(span, max_distance), 1.0);
        // The real per-unit cost is the ground distance raised to p.  (p == 1
        // → cost_bound, leaving the legacy p == 1 scale bit-for-bit unchanged.)
        const double max_real_cost = std::pow(cost_bound, p);
        const double overflow_cap = max_int / (max_real_cost * total);
        const double target = FINE_GRID_TARGET_FLOW / total;
        return std::max(1.0, std::min(target, overflow_cap));
    }

    double sf_distance_ = 0.0;
    double sf_intensity_ = 0.0;
    DistanceMetric metric_;
};

#endif // WNET_SCALING_HPP
