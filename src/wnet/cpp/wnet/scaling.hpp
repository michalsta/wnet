#ifndef WNET_SCALING_HPP
#define WNET_SCALING_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
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
        double max_int = static_cast<double>(int64_t(1) << 60)
    ) : metric_(metric) {
        if (trash_costs.empty())
            throw std::invalid_argument("Scaler: at least one trash cost is required.");

        const double empirical_sum = empirical.sum_intensities();
        double theoretical_sum = 0.0;
        for (const auto* t : theoretical) theoretical_sum += t->sum_intensities();
        const double max_sum = std::max(empirical_sum, theoretical_sum);
        if (max_sum <= 0.0)
            throw std::invalid_argument(
                "Scaler: max intensity sum must be positive (are all spectra empty?).");

        double max_cost = max_distance;
        double min_cost = max_distance;
        for (double c : trash_costs) {
            max_cost = std::max(max_cost, c);
            min_cost = std::min(min_cost, c);
        }
        if (max_cost <= 0.0 || min_cost <= 0.0)
            throw std::invalid_argument(
                "Scaler: cost per unit flow (max_distance / trash costs) must be positive.");

        const bool auto_mode = (explicit_scale_factor <= 0.0) && !tie_factors;

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
        } else {
            // wnetdeconv mode: independent precision-driven factors, capped.
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

        // Distance-resolution guard (auto mode only, mirroring wnetdeconv): the
        // smallest cost-per-unit-flow must survive integer quantization.
        if (enforce_distance_resolution && auto_mode
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
        // max_dropped_fraction >= 1.0 disables the guard.
        if (max_dropped_fraction < 1.0) {
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
    double sf_distance_ = 0.0;
    double sf_intensity_ = 0.0;
    DistanceMetric metric_;
};

#endif // WNET_SCALING_HPP
