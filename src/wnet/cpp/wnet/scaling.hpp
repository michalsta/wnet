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

// =============================================================================
// ScalerBase<DIM> — shared interface and static helper utilities
// =============================================================================
//
// Pure advisor: computes sf_distance_ / sf_intensity_ at construction time and
// exposes them.  The caller applies them to the network (scale positions,
// set_intensity_scale, unscale total_cost).  Construction is the entire failure
// surface — guards throw std::invalid_argument (→ Python ValueError).
//
// Subclasses set sf_distance_ and sf_intensity_ directly and call
// validate_factors() before returning from their constructor.  All helpers are
// static so they can be called without an object.
template<size_t DIM>
class ScalerBase {
public:
    using VecDist = VectorDistribution<DIM, double, double>;

    double sf_distance() const { return sf_distance_; }
    double sf_intensity() const { return sf_intensity_; }
    // Geometric mean: scale_factor**2 == sf_distance * sf_intensity.
    double scale_factor() const { return std::sqrt(sf_distance_ * sf_intensity_); }
    double ftol()         const { return 1.0 / (sf_distance_ * sf_intensity_); }
    DistanceMetric metric() const { return metric_; }

protected:
    double sf_distance_ = 1.0;
    double sf_intensity_ = 1.0;
    DistanceMetric metric_ = DistanceMetric::L2;

    // ---- Validation ---------------------------------------------------------

    static void validate_positive_costs(double max_cost, double min_cost) {
        if (max_cost <= 0.0 || min_cost <= 0.0)
            throw std::invalid_argument(
                "Scaler: cost per unit flow (max_distance / trash costs) must be positive.");
    }

    static void validate_max_sum(double max_sum) {
        if (max_sum <= 0.0)
            throw std::invalid_argument(
                "Scaler: max intensity sum must be positive (are all spectra empty?).");
    }

    void validate_factors() const {
        if (!(sf_distance_ > 0.0 && sf_intensity_ > 0.0))
            throw std::invalid_argument(
                "Scaler: could not compute positive scale factors; check your "
                "data, trash costs, or pass an explicit scale_factor.");
    }

    // ---- Cost / intensity aggregates ----------------------------------------

    // {empirical_sum, sum-of-all-theoretical, max(empirical, theoretical_sum)}.
    static std::tuple<double, double, double> intensity_sums(
        const VecDist& empirical, const std::vector<const VecDist*>& theoretical)
    {
        const double emp = empirical.sum_intensities();
        double theo = 0.0;
        for (const auto* t : theoretical) theo += t->sum_intensities();
        return {emp, theo, std::max(emp, theo)};
    }

    // {max_cost, min_cost} from max_distance and optional trash costs.
    static std::pair<double, double> cost_bounds(
        double max_distance, const std::vector<double>& trash_costs)
    {
        double max_c = max_distance, min_c = max_distance;
        for (double c : trash_costs) {
            max_c = std::max(max_c, c);
            min_c = std::min(min_c, c);
        }
        return {max_c, min_c};
    }

    // int64 overflow cap: sf_intensity * max_cost * max_sum must not exceed max_int.
    static double overflow_cap(double max_cost, double max_sum, double max_int) {
        return max_int / (max_cost * max_sum);
    }

    // ---- Per-spectrum intensity statistics ----------------------------------

    // "Quantile peak": sort intensities descending, return the intensity value
    // at which the running cumulative sum first reaches or exceeds frac*total.
    // Concretely: the least-intense peak still inside the top-frac mass band.
    static double quantile_peak(const VecDist& d, double frac) {
        const double total = d.sum_intensities();
        if (!(total > 0.0)) return 0.0;
        std::vector<double> sorted;
        for (double v : d.get_intensities())
            if (v > 0.0) sorted.push_back(v);
        std::sort(sorted.begin(), sorted.end(), std::greater<double>());
        double cumsum = 0.0;
        for (double v : sorted) {
            cumsum += v;
            if (cumsum >= frac * total) return v;
        }
        return sorted.empty() ? 0.0 : sorted.back();
    }

    // Minimum quantile_peak(frac) across empirical and all theoretical
    // spectra.  Spectra with no positive intensity (quantile_peak == 0, e.g.
    // an empty component after trimming/filtering) contribute nothing to
    // rounding loss and are skipped — they must not zero out the minimum.
    // Returns 0.0 only when ALL spectra are empty/zero.
    static double min_quantile_peak(
        const VecDist& empirical, const std::vector<const VecDist*>& theoretical,
        double frac)
    {
        double p = std::numeric_limits<double>::infinity();
        auto consider = [&](const VecDist& d) {
            const double q = quantile_peak(d, frac);
            if (q > 0.0) p = std::min(p, q);
        };
        consider(empirical);
        for (const auto* t : theoretical)
            consider(*t);
        return std::isinf(p) ? 0.0 : p;
    }

    // ---- Rounding-loss guard ------------------------------------------------

    // Fraction of a spectrum's total intensity lost to round-toward-zero
    // quantisation: (sum - sum(trunc(v*sf))/sf) / sum.
    static double rounding_loss_frac(const VecDist& d, double sf) {
        const double total = d.sum_intensities();
        if (!(total > 0.0)) return 0.0;
        double kept = 0.0;
        for (double v : d.get_intensities())
            kept += std::trunc(v * sf);
        kept /= sf;
        return (total - kept) / total;
    }

    // Throws if any spectrum's rounding loss exceeds max_dropped_fraction.
    static void check_rounding_loss(
        const VecDist& empirical, const std::vector<const VecDist*>& theoretical,
        double sf, double max_dropped_fraction)
    {
        double worst_frac = 0.0;
        std::string worst_name;
        auto check_one = [&](const VecDist& d, const std::string& name) {
            const double frac = rounding_loss_frac(d, sf);
            if (frac > worst_frac) { worst_frac = frac; worst_name = name; }
        };
        check_one(empirical, "empirical_spectrum");
        for (size_t i = 0; i < theoretical.size(); ++i)
            check_one(*theoretical[i], "theoretical_spectra[" + std::to_string(i) + "]");
        if (worst_frac > max_dropped_fraction)
            throw std::invalid_argument(
                "Integer intensity quantization at sf_intensity=" +
                std::to_string(sf) + " would lose " +
                std::to_string(worst_frac * 100.0) + "% of " + worst_name +
                "'s total intensity to rounding (limit " +
                std::to_string(max_dropped_fraction * 100.0) + "%): the "
                "intensity scale is too coarse for this spectrum.  "
                "Pass a larger scale_factor or allow the loss.");
    }

    // ---- Fine-grid intensity helper (WassersteinNetwork/Distance path) ------

    // Maps real intensities onto a fine integer supply grid targeting ~2^30
    // total flow without overflowing the int64 cost accumulator.
    // Returns 1.0 when all intensities are already integer-valued.
    // trash_costs participate in the accumulator bound: they are already in
    // cost (W_p^p) units and a large trash cost (e.g. 1e10) is often the
    // dominant per-unit cost, so ignoring it here would size a grid whose
    // trash flow overflows int64 at solve time.
    static double fine_grid_intensity_scale(
        const VecDist& empirical,
        const std::vector<const VecDist*>& theoretical,
        double p, double max_distance, double max_int,
        const std::vector<double>& trash_costs = {})
    {
        static constexpr double FINE_GRID_TARGET_FLOW =
            static_cast<double>(int64_t(1) << 30);

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
        double max_real_cost = std::pow(cost_bound, p);
        for (double t : trash_costs)          // already in cost (W_p^p) units
            max_real_cost = std::max(max_real_cost, t);
        const double cap = max_int / (max_real_cost * total);
        const double target = FINE_GRID_TARGET_FLOW / total;
        return std::max(1.0, std::min(target, cap));
    }
};


// =============================================================================
// WNetAlignScaler<DIM>
// =============================================================================
//
// Tied single-factor mode used by wnetalign: sf = sqrt(max_int / (max_sum *
// max_cost)), applied equally to both positions and intensities.  No rounding
// guard — the caller pre-scales positions using this sf.
template<size_t DIM>
class WNetAlignScaler : public ScalerBase<DIM> {
public:
    using VecDist = typename ScalerBase<DIM>::VecDist;

    WNetAlignScaler(
        const VecDist& empirical,
        const std::vector<const VecDist*>& theoretical,
        DistanceMetric metric,
        double max_distance,
        const std::vector<double>& trash_costs,
        double max_int = static_cast<double>(int64_t(1) << 60))
    {
        this->metric_ = metric;

        auto [max_c, min_c] = ScalerBase<DIM>::cost_bounds(max_distance, trash_costs);
        ScalerBase<DIM>::validate_positive_costs(max_c, min_c);

        auto [emp_sum, theo_sum, max_sum] =
            ScalerBase<DIM>::intensity_sums(empirical, theoretical);
        ScalerBase<DIM>::validate_max_sum(max_sum);

        const double product = max_sum * max_c;
        if (std::isinf(product))
            throw std::overflow_error(
                "WNetAlignScaler: max_sum * max_cost overflows double.");
        const double f = std::sqrt(max_int / product);
        this->sf_distance_ = f;
        this->sf_intensity_ = f;
        this->validate_factors();
    }
};


// =============================================================================
// WNetDeconvScaler<DIM>
// =============================================================================
//
// Intensity-only scale (sf_distance == 1) anchored to the p95 peak: the peak
// at which the cumulative mass (sorted most→least intense) first crosses 95 %
// of the spectrum's total.  Scale is set so that this peak and all larger ones
// incur at most 10 % relative rounding error.  The bottom 5 % tail rounds more
// freely; the rounding-loss guard is loosened accordingly (default 0.20).
// sf_distance is always 1.0 — the network handles cost quantisation itself.
template<size_t DIM>
class WNetDeconvScaler : public ScalerBase<DIM> {
public:
    using VecDist = typename ScalerBase<DIM>::VecDist;

    WNetDeconvScaler(
        const VecDist& empirical,
        const std::vector<const VecDist*>& theoretical,
        DistanceMetric metric,
        double max_distance,
        const std::vector<double>& trash_costs,
        double max_int             = static_cast<double>(int64_t(1) << 60),
        double max_dropped_fraction = 0.20)
    {
        this->metric_ = metric;

        auto [max_c, min_c] = ScalerBase<DIM>::cost_bounds(max_distance, trash_costs);
        ScalerBase<DIM>::validate_positive_costs(max_c, min_c);

        auto [emp_sum, theo_sum, max_sum] =
            ScalerBase<DIM>::intensity_sums(empirical, theoretical);
        ScalerBase<DIM>::validate_max_sum(max_sum);

        const double p95 = ScalerBase<DIM>::min_quantile_peak(
            empirical, theoretical, 0.95);
        if (!(p95 > 0.0))
            throw std::invalid_argument(
                "WNetDeconvScaler: p95 peak is zero; all spectra appear empty.");

        this->sf_distance_ = 1.0;
        this->sf_intensity_ = 1.0 / (0.10 * p95);
        const double cap = ScalerBase<DIM>::overflow_cap(max_c, max_sum, max_int);
        if (this->sf_intensity_ > cap) this->sf_intensity_ = cap;
        this->validate_factors();

        if (max_dropped_fraction < 1.0)
            ScalerBase<DIM>::check_rounding_loss(
                empirical, theoretical, this->sf_intensity_, max_dropped_fraction);
    }
};


// =============================================================================
// FineGridScaler<DIM>
// =============================================================================
//
// For WassersteinNetwork / WassersteinDistance: sf_distance == 1 (positions
// remain as real ground distances), intensities mapped onto a fine integer grid
// targeting ~2^30 total flow.  Follows warn-not-throw for under-resolution.
template<size_t DIM>
class FineGridScaler : public ScalerBase<DIM> {
public:
    using VecDist = typename ScalerBase<DIM>::VecDist;

    FineGridScaler(
        const VecDist& empirical,
        const std::vector<const VecDist*>& theoretical,
        DistanceMetric metric,
        double max_distance,
        const std::vector<double>& trash_costs,
        double p       = 1.0,
        double max_int = static_cast<double>(int64_t(1) << 60))
    {
        this->metric_ = metric;
        auto [max_c, min_c] = ScalerBase<DIM>::cost_bounds(max_distance, trash_costs);
        ScalerBase<DIM>::validate_positive_costs(max_c, min_c);
        // fine_grid_intensity_scale degrades gracefully to 1.0 for empty spectra.
        this->sf_distance_ = 1.0;
        this->sf_intensity_ = ScalerBase<DIM>::fine_grid_intensity_scale(
            empirical, theoretical, p, max_distance, max_int, trash_costs);
        this->validate_factors();
    }
};


// =============================================================================
// GenericScaler<DIM>
// =============================================================================
//
// New scaler with no backward-compatibility baggage.  Uses the same p-quantile
// intensity policy as WNetDeconvScaler but exposes p95_frac and rounding_tol
// as constructor parameters, making it suitable for any context.
// sf_distance is always 1.0 — cost quantisation belongs to the network.
template<size_t DIM>
class GenericScaler : public ScalerBase<DIM> {
public:
    using VecDist = typename ScalerBase<DIM>::VecDist;

    // p95_frac         : mass fraction defining the "signal" band, in (0, 1].
    // rounding_tol     : max relative rounding error on the quantile peak, in (0, 1].
    // max_dropped_frac : total-rounding-loss guard; >= 1.0 disables it.
    GenericScaler(
        const VecDist& empirical,
        const std::vector<const VecDist*>& theoretical,
        DistanceMetric metric,
        double max_distance,
        const std::vector<double>& trash_costs,
        double p95_frac          = 0.95,
        double rounding_tol      = 0.10,
        double max_dropped_frac  = 0.20,
        double max_int           = static_cast<double>(int64_t(1) << 60))
    {
        this->metric_ = metric;

        if (!(p95_frac > 0.0 && p95_frac <= 1.0))
            throw std::invalid_argument(
                "GenericScaler: p95_frac must be in (0, 1].");
        if (!(rounding_tol > 0.0 && rounding_tol <= 1.0))
            throw std::invalid_argument(
                "GenericScaler: rounding_tol must be in (0, 1].");

        auto [max_c, min_c] = ScalerBase<DIM>::cost_bounds(max_distance, trash_costs);
        ScalerBase<DIM>::validate_positive_costs(max_c, min_c);

        auto [emp_sum, theo_sum, max_sum] =
            ScalerBase<DIM>::intensity_sums(empirical, theoretical);
        ScalerBase<DIM>::validate_max_sum(max_sum);

        const double pq = ScalerBase<DIM>::min_quantile_peak(
            empirical, theoretical, p95_frac);
        if (!(pq > 0.0))
            throw std::invalid_argument(
                "GenericScaler: quantile peak is zero; all spectra appear empty.");

        this->sf_distance_ = 1.0;
        this->sf_intensity_ = 1.0 / (rounding_tol * pq);
        const double cap = ScalerBase<DIM>::overflow_cap(max_c, max_sum, max_int);
        if (this->sf_intensity_ > cap) this->sf_intensity_ = cap;
        this->validate_factors();

        if (max_dropped_frac < 1.0)
            ScalerBase<DIM>::check_rounding_loss(
                empirical, theoretical, this->sf_intensity_, max_dropped_frac);
    }
};


#endif // WNET_SCALING_HPP
