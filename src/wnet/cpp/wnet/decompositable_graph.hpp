#ifndef WNET_DECOMPOSITABLE_GRAPH_HPP
#define WNET_DECOMPOSITABLE_GRAPH_HPP

#include <vector>
#include <span>
#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <optional>
#include <deque>
#include <queue>
#include <variant>
#include <type_traits>


#define LEMON_ONLY_TEMPLATES
#include <lemon/static_graph.h>
#include <lemon/network_simplex.h>
#include <lemon/cycle_canceling.h>
#include <lemon/cost_scaling.h>
#include <lemon/capacity_scaling.h>
#include <pylmcf/network_simplex_lct_adapter.h>

// Pivot rule for NetworkSimplex. Values match lemon::NetworkSimplex::PivotRule.
enum class NSPivotRule {
    FIRST_ELIGIBLE, BEST_ELIGIBLE, BLOCK_SEARCH, CANDIDATE_LIST, ALTERING_LIST
};

// Internal method for CostScaling. Values match lemon::CostScaling::Method.
enum class CSMethod { PUSH, AUGMENT, PARTIAL_AUGMENT };

// Internal method for CycleCanceling. Values match lemon::CycleCanceling::Method.
enum class CCMethod {
    SIMPLE_CYCLE_CANCELING, MINIMUM_MEAN_CYCLE_CANCELING, CANCEL_AND_TIGHTEN
};

// Warm-restart strategy for NetworkSimplex across successive solves:
//   None       - always cold init() (no basis reuse)
//   Simple     - reuse the basis via repairTreeFlows(); cold-fallback if it fails
//   Dual       - Simple, plus a bounded dual-simplex repair before cold-fallback
//   Primal     - Simple, plus a bounded primal-pivot repair before cold-fallback
//   DualRatio  - Dual with bound-flipping long step; fewer cold-fallbacks on
//                large/hard graphs; default choice across most dataset families
//   DualGreedy - Like DualRatio but enters the max-capacity arc (not min-|rc|);
//                may fix violations in fewer pivots when capacities vary widely
//   LinkCut    - EXPERIMENTAL alternative backend: pylmcf::NetworkSimplexLCT
//                (link-cut-tree spanning tree, O(log n) pivot tree updates).
//                Currently Simple-strategy warm only (repair-or-cold); selects
//                a different solver implementation, not a repair strategy.
enum class NSWarmMode { None, Simple, Dual, Primal, DualRatio, DualGreedy, LinkCut };

struct NetworkSimplexConfig {
    NSPivotRule pivot = NSPivotRule::BLOCK_SEARCH;
    NSWarmMode warm = NSWarmMode::DualRatio;
};
struct CostScalingConfig {
    CSMethod method = CSMethod::PARTIAL_AUGMENT;
    int factor = 16;
};
struct CycleCancelingConfig {
    CCMethod method = CCMethod::CANCEL_AND_TIGHTEN;
};
struct CapacityScalingConfig {
    int factor = 4;
};

using SolverConfig = std::variant<
    NetworkSimplexConfig, CostScalingConfig,
    CycleCancelingConfig, CapacityScalingConfig>;

//#include "pylmcf/py_support.h"
#include "graph_elements.hpp"
#include "distribution.hpp"
#include "distances.hpp"

#include <iostream>
#include <cstdint>

// ---------------------------------------------------------------------------
// Cost scaling for order-p (Lp) Wasserstein.
//
// Edges carry their *real* (unscaled, possibly fractional) cost as a double.
// For p != 1 the cost d^p is fractional, so truncating straight to int64 would
// destroy precision for small distances.  Instead we pick an integer scale S
// from the global maximum real cost and store round(S * real) in the solver's
// integer cost map, undoing /S at the public boundary.  p == 1 is special-cased
// to S == 1 and plain truncation, reproducing the legacy integer behaviour
// bit-for-bit.
// ---------------------------------------------------------------------------

// Choose the integer cost scale S subject to two ceilings:
//
//  * per-edge: S * c_max <= 2^52, so every round(S*cost) lands inside the
//    double exact-integer range (round and the /S unscale are both exact) and
//    well below the NetworkSimplex potential-overflow ceiling (ART_COST=2^62).
//
//  * accumulator: the solver sums sum(scaled_cost * flow) in int64, and the
//    total flow is bounded by the total intensity, so S * c_max * total_flow
//    must stay within the int64 accumulator.  Bounding only the per-edge cost
//    (the old behaviour) silently overflowed totalCost() once total_flow grew
//    past ~2^11 — e.g. p=2 over a single peak-pair of mass 1e4 wrapped to a
//    negative total, yielding a complex distance after the ^(1/p) root.
//
// S is the floor of the tighter of the two ceilings (>= 1).  total_flow <= 0
// (unknown / p==1) falls back to the per-edge bound alone.
inline int64_t pick_cost_scale(double c_max, double total_flow, bool p_is_one) {
    if (p_is_one || !(c_max > 0.0)) return 1;
    constexpr double PER_EDGE_TARGET = 4503599627370496.0;   // 2^52
    constexpr double ACCUMULATOR_TARGET = 4611686018427387904.0; // 2^62
    double s = std::floor(PER_EDGE_TARGET / c_max);
    if (total_flow > 0.0) {
        const double s_acc = std::floor(ACCUMULATOR_TARGET / (c_max * total_flow));
        if (s_acc < s) s = s_acc;
    }
    return s >= 1.0 ? static_cast<int64_t>(s) : 1;
}

// Quantise a real edge cost to the solver's integer cost type.
// p == 1: truncate the raw value (legacy, exact).  p != 1: round(S * real).
template <typename VALUE_TYPE>
inline VALUE_TYPE quantize_cost(double real_cost, int64_t scale, bool p_is_one) {
    const double scaled = p_is_one ? real_cost : static_cast<double>(scale) * real_cost;
    if (!std::isfinite(scaled) ||
        scaled > static_cast<double>(std::numeric_limits<VALUE_TYPE>::max()))
        throw std::overflow_error(
            "quantize_cost: scaled cost " + std::to_string(scaled) +
            " overflows the solver cost type (max " +
            std::to_string(std::numeric_limits<VALUE_TYPE>::max()) + ")");
    return p_is_one ? static_cast<VALUE_TYPE>(scaled)
                    : static_cast<VALUE_TYPE>(std::llround(scaled));
}


template <typename VALUE_TYPE, typename intensity_type>
class WassersteinNetworkSubgraph {
    std::vector<FlowNode<intensity_type>> nodes;
    std::vector<FlowEdge<intensity_type>> edges;
    lemon::StaticDigraph lemon_graph;
    lemon::StaticDigraph::NodeMap<VALUE_TYPE> node_supply_map;
    lemon::StaticDigraph::ArcMap<VALUE_TYPE> capacities_map;
    lemon::StaticDigraph::ArcMap<VALUE_TYPE> costs_map;
    std::optional<lemon::NetworkSimplex<lemon::StaticDigraph, VALUE_TYPE, VALUE_TYPE>> ns_solver;
    std::optional<pylmcf::NetworkSimplexLCTAdapter<lemon::StaticDigraph, VALUE_TYPE, VALUE_TYPE>> ns_lct_solver;
    std::optional<lemon::CycleCanceling<lemon::StaticDigraph, VALUE_TYPE, VALUE_TYPE>> cc_solver;
    std::optional<lemon::CostScaling<lemon::StaticDigraph, VALUE_TYPE, VALUE_TYPE>> cs_solver;
    std::optional<lemon::CapacityScaling<lemon::StaticDigraph, VALUE_TYPE, VALUE_TYPE>> cap_solver;
    SolverConfig _config = NetworkSimplexConfig{};
    LEMON_INDEX simple_trash_idx;
    bool simple_trash_added = false;
    bool experimental_trash_added = false;
    bool theoretical_trash_added = false;
    VALUE_TYPE lemon_empirical_intensity;
    VALUE_TYPE lemon_theoretical_intensity;
    const size_t no_target_distributions;
    bool built = false;
    int _cold_starts_via_run = 0;

    // Cost scaling (set by the owning network at build): integer cost map holds
    // quantize_cost(real, _scale, _p_is_one).  _p_is_one => legacy S=1 truncation.
    int64_t _scale = 1;
    bool _p_is_one = true;
    // Intensity scaling (set by the owning network at build): real (double) node
    // intensities are mapped to integer LEMON supplies/capacities as
    // round-toward-zero(real * _intensity_scale).  1.0 reproduces the legacy
    // behaviour (intensities consumed verbatim, truncated to the integer type).
    double _intensity_scale = 1.0;

    // Cached residual/derivative context.  The context is a pure function of
    // the post-solve solver state (potentials, flow, capacities, costs),
    // which only changes via _run_solver(); _solution_version is bumped
    // there, so a cached context whose stamp matches is bit-identical to a
    // fresh recompute (value-exact memoization across multiple derivative
    // queries / identical re-solves on the same solution).  Two independent
    // slots: [0] = exact residual, [1] = fast dual-pi approximation, so the
    // same solved solution can be queried in either form without thrash.
    struct DerivContext {
        VALUE_TYPE INF;
        bool supply_fixed;
        bool asymmetric;
        VALUE_TYPE trash_cost, src_adjust, sink_adjust;
        std::vector<VALUE_TYPE> dist_src, dist_sink;
        std::vector<VALUE_TYPE> theo_sink_slack;
    };
    std::uint64_t _solution_version = 0;
    mutable std::uint64_t _deriv_ctx_version[2] = {
        std::numeric_limits<std::uint64_t>::max(),
        std::numeric_limits<std::uint64_t>::max()};
    mutable DerivContext _deriv_ctx_cache[2];

    struct MatchingEdgeInfo {
        lemon::StaticDigraph::Arc arc;
        intensity_type emp_intensity;
        intensity_type theo_intensity;
        size_t spectrum_id;
        LEMON_INDEX emp_peak_index;
        LEMON_INDEX theo_peak_index;
    };
    struct TheoSinkEdgeInfo {
        lemon::StaticDigraph::Arc arc;
        intensity_type theo_intensity;
        size_t spectrum_id;
    };
    std::vector<MatchingEdgeInfo> _matching_edge_cache;
    std::vector<TheoSinkEdgeInfo> _theo_sink_edge_cache;
    std::vector<uint8_t> _unlimited_arc;  // true for MatchingEdge and ChainEdge
    std::vector<VALUE_TYPE> _costs_buf;   // reusable scratch for update_positions_and_solve
    mutable std::vector<VALUE_TYPE> _chain_R_buf;    // K-1; reused for R/c_right in chain functions
    mutable std::vector<VALUE_TYPE> _chain_L_buf;    // K-1; reused for L/c_left in chain functions
    mutable std::vector<VALUE_TYPE> _chain_dist_buf; // n nodes; reused by chain_residual_distances
    std::vector<double> _chain_pos_buf;              // K; reused by update_positions_and_solve
    mutable std::vector<uint8_t>    _chain_has_src_fwd,  _chain_has_src_rev;
    mutable std::vector<uint8_t>    _chain_has_sink_fwd, _chain_has_sink_rev;
    mutable std::vector<VALUE_TYPE> _chain_exp_trash_cost, _chain_theo_trash_cost;
    mutable std::vector<uint8_t>    _chain_exp_trash_fwd, _chain_exp_trash_rev;
    mutable std::vector<uint8_t>    _chain_theo_trash_fwd, _chain_theo_trash_rev;

    struct ChainTopology {
        std::vector<LEMON_INDEX> order;
        std::vector<LEMON_INDEX> right_arc_ids;
        std::vector<LEMON_INDEX> left_arc_ids;
        std::vector<VALUE_TYPE>  gap_cost;
        std::vector<size_t> node_to_pos;
    };
    std::optional<ChainTopology> _chain_topo;

    // True when the NetworkSimplex backend is the experimental link-cut-tree
    // implementation (NSWarmMode::LinkCut) rather than LEMON's array solver.
    bool _use_lct() const {
        return std::holds_alternative<NetworkSimplexConfig>(_config) &&
               std::get<NetworkSimplexConfig>(_config).warm == NSWarmMode::LinkCut;
    }
    VALUE_TYPE _solver_potential(lemon::StaticDigraph::Node nd) const {
        return _use_lct() ? ns_lct_solver->potential(nd)
                          : ns_solver->potential(nd);
    }
    bool _solver_has_value() const {
        if (std::holds_alternative<NetworkSimplexConfig>(_config))
            return _use_lct() ? ns_lct_solver.has_value() : ns_solver.has_value();
        if (std::holds_alternative<CycleCancelingConfig>(_config)) return cc_solver.has_value();
        if (std::holds_alternative<CostScalingConfig>(_config))    return cs_solver.has_value();
        return cap_solver.has_value();
    }
    VALUE_TYPE _solver_flow(lemon::StaticDigraph::Arc arc) const {
        if (std::holds_alternative<NetworkSimplexConfig>(_config))
            return _use_lct() ? ns_lct_solver->flow(arc) : ns_solver->flow(arc);
        if (std::holds_alternative<CycleCancelingConfig>(_config)) return cc_solver->flow(arc);
        if (std::holds_alternative<CostScalingConfig>(_config))    return cs_solver->flow(arc);
        return cap_solver->flow(arc);
    }
    VALUE_TYPE _solver_total_cost() const {
        if (std::holds_alternative<NetworkSimplexConfig>(_config))
            return _use_lct() ? ns_lct_solver->totalCost() : ns_solver->totalCost();
        if (std::holds_alternative<CycleCancelingConfig>(_config)) return cc_solver->totalCost();
        if (std::holds_alternative<CostScalingConfig>(_config))    return cs_solver->totalCost();
        return cap_solver->totalCost();
    }

    void _build_chain_topology() {
        std::vector<std::vector<std::pair<LEMON_INDEX, LEMON_INDEX>>> chain_adj(nodes.size());
        bool any_chain = false;
        for (LEMON_INDEX ii = 0; ii < static_cast<LEMON_INT>(edges.size()); ++ii) {
            if (std::holds_alternative<ChainEdge>(edges[ii].get_type())) {
                const LEMON_INDEX u = edges[ii].get_start_node_id();
                const LEMON_INDEX v = edges[ii].get_end_node_id();
                chain_adj[u].push_back({v, ii});
                any_chain = true;
            }
        }
        if (!any_chain) return;

        LEMON_INDEX start_node = -1;
        for (LEMON_INDEX node_id = 0; node_id < static_cast<LEMON_INT>(chain_adj.size()); ++node_id) {
            if (chain_adj[node_id].size() == 1) { start_node = node_id; break; }
        }
        if (start_node == -1)
            throw std::runtime_error("Chain subgraph has no endpoint — malformed chain.");

        ChainTopology topo;
        topo.order.push_back(start_node);
        LEMON_INDEX prev = -1, curr = start_node;
        while (true) {
            LEMON_INDEX next = -1, out_arc = -1;
            for (const auto& [neighbor, arc_id] : chain_adj[curr]) {
                if (neighbor != prev) { next = neighbor; out_arc = arc_id; break; }
            }
            if (next == -1) break;
            LEMON_INDEX in_arc = -1;
            for (const auto& [neighbor, arc_id] : chain_adj[next]) {
                if (neighbor == curr) { in_arc = arc_id; break; }
            }
            if (in_arc == -1)
                throw std::runtime_error("Chain arc is not bidirectional — malformed chain.");
            topo.order.push_back(next);
            topo.right_arc_ids.push_back(out_arc);
            topo.left_arc_ids.push_back(in_arc);
            topo.gap_cost.push_back(costs_map[lemon_graph.arcFromId(out_arc)]);
            prev = curr; curr = next;
        }
        topo.node_to_pos.assign(nodes.size(), std::numeric_limits<size_t>::max());
        for (size_t i = 0; i < topo.order.size(); ++i)
            topo.node_to_pos[topo.order[i]] = i;
        _chain_topo = std::move(topo);
    }

    // Run (or warm-restart) the configured solver using the current
    // capacities_map, costs_map and node_supply_map.
    // costs_changed=true: costs were updated since the last solve (e.g.
    // after update_positions); for NetworkSimplex the cost map must be
    // re-pushed before warmRun so that the dual variables are consistent.
    void _run_solver(bool costs_changed = false) {
        std::visit([&](const auto& cfg) {
            using T = std::decay_t<decltype(cfg)>;
            if constexpr (std::is_same_v<T, NetworkSimplexConfig>) {
              if (cfg.warm == NSWarmMode::LinkCut) {
                // Experimental link-cut-tree backend (Simple warm strategy:
                // repair-or-cold).  pivot/strategy are not applicable.
                if (ns_lct_solver.has_value()) {
                    ns_lct_solver->upperMap(capacities_map);
                    ns_lct_solver->supplyMap(node_supply_map);
                    if (costs_changed) ns_lct_solver->costMap(costs_map);
                    ns_lct_solver->warmRun();
                } else {
                    ++_cold_starts_via_run;
                    ns_lct_solver.emplace(lemon_graph);
                    ns_lct_solver->upperMap(capacities_map);
                    ns_lct_solver->costMap(costs_map);
                    ns_lct_solver->supplyMap(node_supply_map);
                    ns_lct_solver->run();
                }
              } else {
                using LemonPR = lemon::NetworkSimplex<lemon::StaticDigraph, VALUE_TYPE, VALUE_TYPE>::PivotRule;
                const auto pivot = static_cast<LemonPR>(cfg.pivot);
                if (cfg.warm != NSWarmMode::None && ns_solver.has_value()) {
                    // Warm start: reuse the existing solver and its spanning-tree
                    // basis.  Only capacities and supplies change between calls
                    // (costs are fixed at build() time), so warmRun() can repair
                    // the previous optimal basis and reach the new optimum with
                    // far fewer pivots.  Dual/Primal modes additionally attempt
                    // a dual-simplex / primal-pivot repair before any cold
                    // fallback; all modes fall back to a cold start if the
                    // basis cannot be repaired.
                    using LemonWR = lemon::NetworkSimplex<lemon::StaticDigraph, VALUE_TYPE, VALUE_TYPE>::WarmRepair;
                    const LemonWR strategy =
                        cfg.warm == NSWarmMode::Dual       ? LemonWR::Dual       :
                        cfg.warm == NSWarmMode::Primal     ? LemonWR::Primal     :
                        cfg.warm == NSWarmMode::DualRatio  ? LemonWR::DualRatio  :
                        cfg.warm == NSWarmMode::DualGreedy ? LemonWR::DualGreedy :
                                                             LemonWR::RepairOnly;
                    ns_solver->upperMap(capacities_map);
                    ns_solver->supplyMap(node_supply_map);
                    if (costs_changed) ns_solver->costMap(costs_map);
                    ns_solver->warmRun(pivot, strategy);
                } else {
                    ++_cold_starts_via_run;
                    ns_solver.emplace(lemon_graph);
                    ns_solver->upperMap(capacities_map);
                    ns_solver->costMap(costs_map);
                    ns_solver->supplyMap(node_supply_map);
                    ns_solver->run(pivot);
                }
              }
            } else if constexpr (std::is_same_v<T, CycleCancelingConfig>) {
                using LemonM = lemon::CycleCanceling<lemon::StaticDigraph, VALUE_TYPE, VALUE_TYPE>::Method;
                cc_solver.emplace(lemon_graph);
                cc_solver->upperMap(capacities_map);
                cc_solver->costMap(costs_map);
                cc_solver->supplyMap(node_supply_map);
                cc_solver->run(static_cast<LemonM>(cfg.method));
            } else if constexpr (std::is_same_v<T, CostScalingConfig>) {
                using LemonM = lemon::CostScaling<lemon::StaticDigraph, VALUE_TYPE, VALUE_TYPE>::Method;
                cs_solver.emplace(lemon_graph);
                cs_solver->upperMap(capacities_map);
                cs_solver->costMap(costs_map);
                cs_solver->supplyMap(node_supply_map);
                cs_solver->run(static_cast<LemonM>(cfg.method), cfg.factor);
            } else {
                static_assert(std::is_same_v<T, CapacityScalingConfig>);
                cap_solver.emplace(lemon_graph);
                cap_solver->upperMap(capacities_map);
                cap_solver->costMap(costs_map);
                cap_solver->supplyMap(node_supply_map);
                cap_solver->run(cfg.factor);
            }
        }, _config);
        // Any solve may have changed potentials/flow -> invalidate the
        // cached derivative context.
        ++_solution_version;
    }

public:
    WassersteinNetworkSubgraph(
        const std::vector<LEMON_INDEX>& subgraph_node_ids,
        const std::vector<FlowNode<intensity_type>>& all_nodes,
        const std::vector<FlowEdge<intensity_type>*>& my_edges,
        size_t no_target_distributions_
    ) :
        lemon_graph(),
        node_supply_map(lemon_graph),
        capacities_map(lemon_graph),
        costs_map(lemon_graph),
        simple_trash_idx(std::numeric_limits<LEMON_INDEX>::max()),
        lemon_empirical_intensity(0),
        lemon_theoretical_intensity(0),
        no_target_distributions(no_target_distributions_)
    {
        nodes.reserve(subgraph_node_ids.size()+2);
        nodes.push_back(FlowNode<intensity_type>(0, SourceNode()));
        nodes.push_back(FlowNode<intensity_type>(1, SinkNode()));
        auto& source_node = nodes[0];
        auto& sink_node = nodes[1];

        std::vector<LEMON_INDEX> node_id_map(all_nodes.size(), -1);

        for (const auto& node_id : subgraph_node_ids)
        {
            node_id_map[node_id] = nodes.size();
            const FlowNodeType<intensity_type>& node_type = all_nodes[node_id].get_type();
            nodes.push_back(FlowNode<intensity_type>(nodes.size(), node_type));
            auto& new_node = nodes.back();
            if(std::holds_alternative<EmpiricalNode<intensity_type>>(node_type))
            {
                edges.emplace_back(
                    edges.size(),
                    source_node,
                    new_node,
                    SrcToEmpiricalEdge()
                );
            }
            else if(std::holds_alternative<TheoreticalNode<intensity_type>>(node_type))
            {
                edges.emplace_back(
                    edges.size(),
                    new_node,
                    sink_node,
                    TheoreticalToSinkEdge()
                );
            }
            else throw std::runtime_error("Invalid FlowNode type. This shouldn't happen.");
        }

        for (const FlowEdge<intensity_type>* edge : my_edges)
        {
            const FlowNode<intensity_type>& start_node = edge->get_start_node();
            const LEMON_INDEX start_local = node_id_map[start_node.get_id()];
            if (start_local == -1) throw std::runtime_error("Start node of edge not found in subgraph nodes.");
            const FlowNode<intensity_type>& end_node = edge->get_end_node();
            const LEMON_INDEX end_local = node_id_map[end_node.get_id()];
            if (end_local == -1) throw std::runtime_error("End node of edge not found in subgraph nodes.");
            edges.emplace_back(
                    edges.size(),
                    nodes[start_local],
                    nodes[end_local],
                    edge->get_type()
            );
        }
    }

    WassersteinNetworkSubgraph(const WassersteinNetworkSubgraph&) = delete;
    WassersteinNetworkSubgraph& operator=(const WassersteinNetworkSubgraph&) = delete;
    WassersteinNetworkSubgraph(WassersteinNetworkSubgraph&&) = delete;
    WassersteinNetworkSubgraph& operator=(WassersteinNetworkSubgraph&&) = delete;

    void add_simple_trash(double cost) {
        if (simple_trash_added)
            throw std::runtime_error("Simple trash edge already added.");
        if (experimental_trash_added || theoretical_trash_added)
            throw std::runtime_error("add_simple_trash() is exclusive with experimental/theoretical trash.");
        if (built)
            throw std::runtime_error("add_simple_trash() must be called before build(), not after.");
        edges.emplace_back(
            edges.size(),
            nodes[0],
            nodes[1],
            SimpleTrashEdge(cost)
        );
        simple_trash_added = true;
    }

    void add_experimental_trash(double cost) {
        if (simple_trash_added)
            throw std::runtime_error("add_experimental_trash() is exclusive with simple trash.");
        if (experimental_trash_added)
            throw std::runtime_error("Experimental trash already added.");
        if (built)
            throw std::runtime_error("add_experimental_trash() must be called before build().");
        // One EmpiricalTrashEdge per empirical node: EmpiricalNode -> Sink.
        // Capacity is set to INF in build(); non-binding (bounded by source supply).
        for (const auto& node : nodes) {
            if (!std::holds_alternative<EmpiricalNode<intensity_type>>(node.get_type())) continue;
            edges.emplace_back(edges.size(), node, nodes[1], EmpiricalTrashEdge(cost));
        }
        experimental_trash_added = true;
    }

    void add_theoretical_trash(double cost) {
        if (simple_trash_added)
            throw std::runtime_error("add_theoretical_trash() is exclusive with simple trash.");
        if (theoretical_trash_added)
            throw std::runtime_error("Theoretical trash already added.");
        if (built)
            throw std::runtime_error("add_theoretical_trash() must be called before build().");
        // One TheoreticalTrashEdge per theoretical node: Source -> TheoreticalNode.
        // Capacity is set to INF in build(); non-binding (bounded by source supply).
        for (const auto& node : nodes) {
            if (!std::holds_alternative<TheoreticalNode<intensity_type>>(node.get_type())) continue;
            edges.emplace_back(edges.size(), nodes[0], node, TheoreticalTrashEdge(cost));
        }
        theoretical_trash_added = true;
    }

    // Real (unscaled) simple-trash cost.  The scaled value used inside the
    // solver/derivatives is costs_map[arc at simple_trash_idx].
    double simple_trash_cost() const {
        if (simple_trash_idx == std::numeric_limits<LEMON_INDEX>::max())
            throw std::runtime_error("Simple trash edge not added.");
        return std::visit([](const auto& arg) -> double {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, SimpleTrashEdge>) return arg.get_cost();
            else { throw std::runtime_error("Invalid FlowEdgeType at simple_trash_idx"); };
        }, edges[simple_trash_idx].get_type());
    }

    void build_impl() {
        assert_fits_lemon_index(nodes.size(), "subgraph nodes");
        assert_fits_lemon_index(edges.size(), "subgraph edges");
        edges = std::move(sorted_copy(edges, [](const FlowEdge<intensity_type>& a, const FlowEdge<intensity_type>& b) {
            if(a.get_start_node_id() != b.get_start_node_id())
                return a.get_start_node_id() < b.get_start_node_id();
            return a.get_end_node_id() < b.get_end_node_id();
        }));
        std::vector<std::pair<LEMON_INDEX, LEMON_INDEX>> arcs;
        arcs.reserve(edges.size());
        for (const FlowEdge<intensity_type>& edge : edges)
            arcs.emplace_back(edge.get_start_node_id(), edge.get_end_node_id());
        lemon_graph.build(nodes.size(), arcs.begin(), arcs.end());

        for (LEMON_INDEX ii = 0; ii < static_cast<LEMON_INT>(nodes.size()); ++ii)
            node_supply_map[lemon_graph.nodeFromId(ii)] = 0;

        for (LEMON_INDEX ii = 0; ii < static_cast<LEMON_INT>(edges.size()); ++ii)
            costs_map[lemon_graph.arcFromId(ii)] = std::visit([&](const auto& arg) -> VALUE_TYPE {
                    using T = std::decay_t<decltype(arg)>;
                    // Cost-bearing edges hold a real (double) cost; quantise it to
                    // the integer solver cost here using the network-wide scale.
                    if constexpr (std::is_same_v<T, MatchingEdge>) return quantize_cost<VALUE_TYPE>(arg.get_cost(), _scale, _p_is_one);
                    else if constexpr (std::is_same_v<T, SrcToEmpiricalEdge>) return (VALUE_TYPE) 0;
                    else if constexpr (std::is_same_v<T, TheoreticalToSinkEdge>) return (VALUE_TYPE) 0;
                    else if constexpr (std::is_same_v<T, SimpleTrashEdge>) { simple_trash_idx = ii; return quantize_cost<VALUE_TYPE>(arg.get_cost(), _scale, _p_is_one); }
                    else if constexpr (std::is_same_v<T, ChainEdge>) return quantize_cost<VALUE_TYPE>(arg.get_cost(), _scale, _p_is_one);
                    else if constexpr (std::is_same_v<T, EmpiricalTrashEdge>) return quantize_cost<VALUE_TYPE>(arg.get_cost(), _scale, _p_is_one);
                    else if constexpr (std::is_same_v<T, TheoreticalTrashEdge>) return quantize_cost<VALUE_TYPE>(arg.get_cost(), _scale, _p_is_one);
                    else { throw std::runtime_error("Invalid FlowEdgeType"); };
                }, edges[ii].get_type());

        for (LEMON_INDEX ii = 0; ii < static_cast<LEMON_INT>(edges.size()); ++ii)
        {
            capacities_map[lemon_graph.arcFromId(ii)] = std::visit([&](const auto& arg) {
                    using T = std::decay_t<decltype(arg)>;
                    if constexpr (std::is_same_v<T, MatchingEdge>) return (VALUE_TYPE) 0;
                    else if constexpr (std::is_same_v<T, SrcToEmpiricalEdge>) {
                        VALUE_TYPE lemon_intensity = static_cast<VALUE_TYPE>(
                            std::get<EmpiricalNode<intensity_type>>(edges[ii].get_end_node().get_type()).get_intensity() * _intensity_scale);
                        lemon_empirical_intensity += lemon_intensity;
                        return lemon_intensity;
                    }
                    else if constexpr (std::is_same_v<T, TheoreticalToSinkEdge>) return (VALUE_TYPE) 0;
                    // Simple trash carries flow up to lemon_total_flow (the supply set on
                    // source/sink). A tight cap = lemon_total_flow is redundant — flow is
                    // already bounded by supply — and causes warm-restart to fail whenever
                    // lemon_total_flow shrinks below the previous trash flow. Use INF so the
                    // cap is non-binding and never needs updating.
                    else if constexpr (std::is_same_v<T, SimpleTrashEdge>) return std::numeric_limits<VALUE_TYPE>::max();
                    // Chain, empirical-trash, and theoretical-trash edges carry unlimited
                    // flow. Use INF (= numeric_limits::max() for int64, which equals LEMON's
                    // INF sentinel). LEMON's findLeavingArc guards c >= MAX → INF so residual
                    // capacity is correctly treated as infinite; LEMON uses this same value
                    // for its own artificial arcs. max/2 was wrong: it bypassed the guard,
                    // returning max/2 - flow (finite) instead of INF.
                    else if constexpr (std::is_same_v<T, ChainEdge>) return std::numeric_limits<VALUE_TYPE>::max();
                    // Asymmetric trash edges: the adjacent anchor edge is always the
                    // binding constraint (SrcToEmpiricalEdge caps empirical inflow;
                    // TheoreticalToSinkEdge caps theoretical outflow), so a redundant
                    // tight cap here adds pivot candidates without shrinking the feasible
                    // region. Use INF like ChainEdge and skip set_point updates.
                    else if constexpr (std::is_same_v<T, EmpiricalTrashEdge>) return std::numeric_limits<VALUE_TYPE>::max();
                    else if constexpr (std::is_same_v<T, TheoreticalTrashEdge>) return std::numeric_limits<VALUE_TYPE>::max();
                    else { throw std::runtime_error("Invalid FlowEdgeType"); };
                }, edges[ii].get_type());
        }
        ns_solver.reset();
        ns_lct_solver.reset();
        cc_solver.reset();
        _build_chain_topology();
        _matching_edge_cache.clear();
        _theo_sink_edge_cache.clear();
        _unlimited_arc.assign(edges.size(), false);
        for (LEMON_INDEX ii = 0; ii < static_cast<LEMON_INT>(edges.size()); ++ii) {
            std::visit([&](const auto& arg) {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, MatchingEdge>) {
                    _unlimited_arc[ii] = true;
                    const auto& theo = std::get<TheoreticalNode<intensity_type>>(edges[ii].get_end_node().get_type());
                    const auto& emp  = std::get<EmpiricalNode<intensity_type>>(edges[ii].get_start_node().get_type());
                    _matching_edge_cache.push_back({lemon_graph.arcFromId(ii), emp.get_intensity(), theo.get_intensity(), theo.get_spectrum_id(), emp.get_peak_index(), theo.get_peak_index()});
                } else if constexpr (std::is_same_v<T, ChainEdge>) {
                    _unlimited_arc[ii] = true;
                } else if constexpr (std::is_same_v<T, TheoreticalToSinkEdge>) {
                    const auto& theo = std::get<TheoreticalNode<intensity_type>>(edges[ii].get_start_node().get_type());
                    _theo_sink_edge_cache.push_back({lemon_graph.arcFromId(ii), theo.get_intensity(), theo.get_spectrum_id()});
                }
            }, edges[ii].get_type());
        }
        _costs_buf.assign(edges.size(), VALUE_TYPE(0));
        if (_chain_topo.has_value()) {
            const size_t K = _chain_topo->order.size();
            const size_t Km1 = K > 1 ? K - 1 : 0;
            _chain_R_buf.resize(Km1);
            _chain_L_buf.resize(Km1);
            _chain_pos_buf.resize(K);
            _chain_dist_buf.resize(nodes.size());
            _chain_has_src_fwd.resize(K);   _chain_has_src_rev.resize(K);
            _chain_has_sink_fwd.resize(K);  _chain_has_sink_rev.resize(K);
            _chain_exp_trash_cost.resize(K); _chain_theo_trash_cost.resize(K);
            _chain_exp_trash_fwd.resize(K); _chain_exp_trash_rev.resize(K);
            _chain_theo_trash_fwd.resize(K); _chain_theo_trash_rev.resize(K);
        }
        built = true;
    }

    // Set by the owning network before build() so build_impl() quantises costs
    // with the network-wide scale.  Kept separate from build() to leave the
    // Python-facing build(config) signature unchanged.
    void set_cost_scaling(int64_t scale, bool p_is_one, double intensity_scale = 1.0) {
        _scale = scale;
        _p_is_one = p_is_one;
        _intensity_scale = intensity_scale;
    }

    void build(SolverConfig config = NetworkSimplexConfig{}) {
        _config = config;
        build_impl();
    }

    void set_point(const std::vector<double>& point) {
        if(point.size() != no_target_distributions)
            throw std::runtime_error("Point dimension: " + std::to_string(point.size()) + " does not match number of target distributions: " + std::to_string(no_target_distributions));
        lemon_theoretical_intensity = 0;
        for (const auto& e : _matching_edge_cache) {
            capacities_map[e.arc] = (VALUE_TYPE) std::min<double>(
                e.theo_intensity * point[e.spectrum_id] * _intensity_scale,
                e.emp_intensity * _intensity_scale);
        }
        for (const auto& e : _theo_sink_edge_cache) {
            VALUE_TYPE lemon_intensity = (VALUE_TYPE) (e.theo_intensity * point[e.spectrum_id] * _intensity_scale);
            capacities_map[e.arc] = lemon_intensity;
            lemon_theoretical_intensity += lemon_intensity;
        }
        // Determine how many units to push from source to sink.
        // When both sides can absorb excess (simple trash, or both asymmetric
        // trash types), use max so every peak participates.  When only one
        // asymmetric trash direction is present, cap supply to the side that
        // has a valid escape route so the MCF is feasible.  No-trash always
        // throws — see the comment inside the else branch.
        VALUE_TYPE lemon_total_flow = 0;
        if (simple_trash_added || (experimental_trash_added && theoretical_trash_added)) {
            lemon_total_flow = std::max<VALUE_TYPE>(lemon_empirical_intensity, lemon_theoretical_intensity);
        } else if (experimental_trash_added) {
            lemon_total_flow = lemon_empirical_intensity;
        } else if (theoretical_trash_added) {
            lemon_total_flow = lemon_theoretical_intensity;
        } else {
            // No trash edges present.  All MCF solvers are unsafe or produce
            // incorrect results without a trash escape route:
            //
            // NetworkSimplex: signed-integer overflow UB in updatePotential().
            //   ART_COST = 2^62; potential accumulation during init pivots can
            //   reach 2^63 — UB.  GCC wraps and terminates; Clang loops forever.
            //
            // CostScaling / CapacityScaling: no ART_COST issue, but
            //   lemon_total_flow = min(emp, theo) is infeasible on sparse graphs
            //   (some units have no matching path), causing these solvers to
            //   return INFEASIBLE with totalCost() = 0 — silently wrong.
            //
            // The correct fix is to add trash edges before calling solve():
            //   use add_simple_trash(cost) to give every unit an escape route.
            // No trash: safe only when supply == demand (empirical == theoretical
            // intensity after integer quantisation).  A balanced, dense-matching
            // network is always feasible; LEMON drives out its artificial arcs
            // immediately so the ART_COST = 2^62 potential-accumulation UB
            // cannot occur.  Verified empirically under ASan+UBSan: 298 tests,
            // zero reports.  Unbalanced or sparse cases must use add_simple_trash().
            if (lemon_empirical_intensity != lemon_theoretical_intensity)
                throw std::runtime_error(
                    "wnet: solve() without trash edges requires equal empirical and "
                    "theoretical intensities (balanced supply/demand). "
                    "Call add_simple_trash() before build() to fix this."
                );
            lemon_total_flow = lemon_empirical_intensity;
        }
        // Trash cap/cost are fixed at build time (cap = INF, cost = SimpleTrashEdge.cost);
        // touching them here would force a warm-restart cold fallback whenever lemon_total_flow
        // changes between solves. Flow on the trash arc is already bounded by source supply.
        node_supply_map[lemon_graph.nodeFromId(0)] = lemon_total_flow;
        node_supply_map[lemon_graph.nodeFromId(1)] = -lemon_total_flow;
        _run_solver();
    }

    VALUE_TYPE total_cost() const {
        if(!_solver_has_value()) throw std::runtime_error("You must call build() and set_point() before calling total_cost().");
        return _solver_total_cost();
    };

    int warm_start_count() const {
        if (_use_lct())
            return ns_lct_solver.has_value() ? ns_lct_solver->warmStartCount() : 0;
        return ns_solver.has_value() ? ns_solver->warmStartCount() : 0;
    }
    int cold_start_count() const {
        if (_use_lct())
            return _cold_starts_via_run +
                   (ns_lct_solver.has_value() ? ns_lct_solver->coldStartCount() : 0);
        return _cold_starts_via_run +
               (ns_solver.has_value() ? ns_solver->coldStartCount() : 0);
    }
    int dual_repair_count() const {
        if (_use_lct())
            return ns_lct_solver.has_value() ? ns_lct_solver->dualRepairCount() : 0;
        return ns_solver.has_value() ? ns_solver->dualRepairCount() : 0;
    }
    int primal_repair_count() const {
        if (_use_lct())
            return ns_lct_solver.has_value() ? ns_lct_solver->primalRepairCount() : 0;
        return ns_solver.has_value() ? ns_solver->primalRepairCount() : 0;
    }


    std::string to_string() const {
        std::string result;
        result += "FlowSubgraph:\n";
        result += "Nodes:\n";
        for (const auto& node : nodes) {
            result += node.to_string() + "\n";
        }
        result += "Edges:\n";
        for (int ii = 0; ii < lemon_graph.arcNum(); ++ii) {
            result += "Edge " + std::to_string(lemon_graph.id(lemon_graph.arcFromId(ii))) + ": " +
                      std::to_string(lemon_graph.id(lemon_graph.source(lemon_graph.arcFromId(ii)))) + " -> " +
                      std::to_string(lemon_graph.id(lemon_graph.target(lemon_graph.arcFromId(ii)))) + " cost: " +
                      std::to_string(costs_map[lemon_graph.arcFromId(ii)]) + " capacity: " +
                      std::to_string(capacities_map[lemon_graph.arcFromId(ii)]) + " flow: " +
                      (_solver_has_value() ?
                      std::to_string(_solver_flow(lemon_graph.arcFromId(ii))) + "\n" :  "not yet computed\n");
        }
        return result;
    };

    std::string lemon_to_string() const {
        std::string result;
        result += "Lemon graph:\n";
        result += "Nodes:\n";
        for (int ii = 0; ii < lemon_graph.nodeNum(); ++ii) {
            result += "Node " + std::to_string(lemon_graph.id(lemon_graph.nodeFromId(ii))) + " supply: " +
                      std::to_string(node_supply_map[lemon_graph.nodeFromId(ii)]) + "\n";
        }
        result += "Edges:\n";
        for (int ii = 0; ii < lemon_graph.arcNum(); ++ii) {
            result += "Edge " + std::to_string(lemon_graph.id(lemon_graph.arcFromId(ii))) + ": " +
                      std::to_string(lemon_graph.id(lemon_graph.source(lemon_graph.arcFromId(ii)))) + " -> " +
                      std::to_string(lemon_graph.id(lemon_graph.target(lemon_graph.arcFromId(ii)))) + " cost: " +
                      std::to_string(costs_map[lemon_graph.arcFromId(ii)]) + " capacity: " +
                      std::to_string(capacities_map[lemon_graph.arcFromId(ii)]) + " flow: " +
                      (_solver_has_value() ?
                      std::to_string(_solver_flow(lemon_graph.arcFromId(ii))) + "\n" :  "not yet computed\n");
        }
        return result;
    };

    size_t no_nodes() const {
        return nodes.size();
    };

    size_t no_edges() const {
        return edges.size();
    };

    const std::vector<FlowNode<intensity_type>>& get_nodes() const {
        return nodes;
    };

    const std::vector<FlowEdge<intensity_type>>& get_edges() const {
        return edges;
    };

    std::vector<VALUE_TYPE>& costs_scratch() { return _costs_buf; }
    std::vector<double>& chain_pos_scratch() { return _chain_pos_buf; }

    void flows_for_target(size_t spectrum_id,
                            std::vector<LEMON_INDEX>& empirical_peak_indices,
                            std::vector<LEMON_INDEX>& theoretical_peak_indices,
                            std::vector<VALUE_TYPE>& flows) const
    {
        for (const auto& e : _matching_edge_cache) {
            if (e.spectrum_id != spectrum_id) continue;
            const VALUE_TYPE flow = _solver_flow(e.arc);
            if (flow == 0) continue;
            empirical_peak_indices.push_back(e.emp_peak_index);
            theoretical_peak_indices.push_back(e.theo_peak_index);
            flows.push_back(flow);
        }
        if (has_chain_edges())
            flows_for_target_chain(
                spectrum_id, empirical_peak_indices,
                theoretical_peak_indices, flows);
    };

    // Sweep-line reconstruction of per-(empirical, theoretical) flows from
    // chain-arc flows. Each chain-edge subgraph holds one linear chain of
    // empirical and theoretical nodes connected by bidirectional ChainEdges.
    // Flow on those arcs encodes transport between pairs without recording
    // which empirical unit ended up at which theoretical. We recover one
    // valid FIFO decomposition in two passes: left-to-right for rightward
    // net flow, right-to-left for leftward net flow. In canonical min-cost
    // solutions, at most one direction carries flow on any positive-cost
    // gap; on zero-cost gaps both directions may be non-zero, which the
    // two-pass split still handles correctly.
    void flows_for_target_chain(
        size_t spectrum_id,
        std::vector<LEMON_INDEX>& empirical_peak_indices,
        std::vector<LEMON_INDEX>& theoretical_peak_indices,
        std::vector<VALUE_TYPE>& flows) const
    {
        if (!_chain_topo.has_value()) return;
        const auto& topo = *_chain_topo;
        const size_t K = topo.order.size();
        if (K < 2) return;  // Isolated node — no gap flow to decompose.

        // Read per-gap forward/reverse flows from the solver.
        std::vector<VALUE_TYPE> R(K - 1), L(K - 1);
        for (size_t g = 0; g < K - 1; ++g) {
            R[g] = _solver_flow(lemon_graph.arcFromId(topo.right_arc_ids[g]));
            L[g] = _solver_flow(lemon_graph.arcFromId(topo.left_arc_ids[g]));
        }

        // Pass 1 — rightward decomposition.
        // delta = R[i] - R[i-1] is the change in the rightward conveyor
        // across node i. delta > 0 means this (empirical) node injects flow
        // onto the conveyor; delta < 0 means this (theoretical) node drains
        // it. In a canonical min-cost flow the sign of delta at a node
        // matches that node's role.
        {
            std::deque<std::pair<LEMON_INDEX, VALUE_TYPE>> queue;
            for (size_t i = 0; i < K; ++i) {
                const VALUE_TYPE r_in = (i == 0) ? 0 : R[i - 1];
                const VALUE_TYPE r_out = (i == K - 1) ? 0 : R[i];
                const VALUE_TYPE delta = r_out - r_in;
                const auto& node_type = nodes[topo.order[i]].get_type();
                if (delta > 0) {
                    const auto* emp = std::get_if<EmpiricalNode<intensity_type>>(&node_type);
                    if (emp == nullptr) continue;
                    queue.push_back({emp->get_peak_index(), delta});
                } else if (delta < 0) {
                    const auto* theo = std::get_if<TheoreticalNode<intensity_type>>(&node_type);
                    if (theo == nullptr) continue;
                    const bool is_target = (theo->get_spectrum_id() == spectrum_id);
                    VALUE_TYPE remaining = -delta;
                    while (remaining > 0 && !queue.empty()) {
                        auto& front = queue.front();
                        const VALUE_TYPE take = std::min(remaining, front.second);
                        if (is_target) {
                            empirical_peak_indices.push_back(front.first);
                            theoretical_peak_indices.push_back(theo->get_peak_index());
                            flows.push_back(take);
                        }
                        remaining -= take;
                        front.second -= take;
                        if (front.second == 0) queue.pop_front();
                    }
                }
            }
        }

        // Pass 2 — leftward decomposition (mirror of pass 1, walking R→L).
        {
            std::deque<std::pair<LEMON_INDEX, VALUE_TYPE>> queue;
            for (size_t ii = 0; ii < K; ++ii) {
                const size_t i = K - 1 - ii;
                const VALUE_TYPE l_in = (i == K - 1) ? 0 : L[i];
                const VALUE_TYPE l_out = (i == 0) ? 0 : L[i - 1];
                const VALUE_TYPE delta = l_out - l_in;
                const auto& node_type = nodes[topo.order[i]].get_type();
                if (delta > 0) {
                    const auto* emp = std::get_if<EmpiricalNode<intensity_type>>(&node_type);
                    if (emp == nullptr) continue;
                    queue.push_back({emp->get_peak_index(), delta});
                } else if (delta < 0) {
                    const auto* theo = std::get_if<TheoreticalNode<intensity_type>>(&node_type);
                    if (theo == nullptr) continue;
                    const bool is_target = (theo->get_spectrum_id() == spectrum_id);
                    VALUE_TYPE remaining = -delta;
                    while (remaining > 0 && !queue.empty()) {
                        auto& front = queue.front();
                        const VALUE_TYPE take = std::min(remaining, front.second);
                        if (is_target) {
                            empirical_peak_indices.push_back(front.first);
                            theoretical_peak_indices.push_back(theo->get_peak_index());
                            flows.push_back(take);
                        }
                        remaining -= take;
                        front.second -= take;
                        if (front.second == 0) queue.pop_front();
                    }
                }
            }
        }
    };

    std::unordered_map<LEMON_INDEX, VALUE_TYPE> get_flow_map() const {
        std::unordered_map<LEMON_INDEX, VALUE_TYPE> result;
        for (LEMON_INDEX ii = 0; ii < static_cast<LEMON_INT>(edges.size()); ++ii)
        {
            const FlowEdge<intensity_type>& edge = edges[ii];
            const VALUE_TYPE flow = _solver_flow(lemon_graph.arcFromId(ii));
            if (flow == 0) continue;
            result[edge.get_id()] = flow;
        }
        return result;
    }

    template<typename T>
    size_t count_nodes_of_type() const {
        size_t result = 0;
        for (const auto& node : nodes)
            if(std::holds_alternative<T>(node.get_type()))
                result++;
        return result;
    }

    template<typename T>
    size_t count_edges_of_type() const {
        size_t result = 0;
        for (const auto& edge : edges)
            if(std::holds_alternative<T>(edge.get_type()))
                result++;
        return result;
    }

    double matching_density() const {
        const double nominator = count_edges_of_type<MatchingEdge>() + count_edges_of_type<ChainEdge>() / 2.0;
        const double denominator = count_nodes_of_type<EmpiricalNode<intensity_type>>() * count_nodes_of_type<TheoreticalNode<intensity_type>>();
        if (denominator == 0) return std::numeric_limits<double>::quiet_NaN();
        return nominator / denominator;
    }

    std::vector<size_t> theoretical_spectra_involved() const {
        std::unique_ptr<bool[]> involved = std::make_unique<bool[]>(no_target_distributions);
        std::fill(involved.get(), involved.get() + no_target_distributions, false);
        for (const auto& node : nodes)
        {
            if (auto node_type = std::get_if<TheoreticalNode<intensity_type>>(&node.get_type()))
            {
                const auto& theoretical_node = *node_type;
                involved[theoretical_node.get_spectrum_id()] = true;
            }
        }
        std::vector<size_t> result;
        for (size_t ii = 0; ii < no_target_distributions; ++ii)
            if(involved[ii])
                result.push_back(ii);
        return result;
    }

    bool is_solved() const {
        return _solver_has_value();
    }

    // Chain-specialized residual shortest distances. Linear sweep variant of
    // bellman_ford_residual for chain subgraphs: rather than relaxing every
    // arc O(n) times, we propagate along the chain (L→R and R→L sweeps)
    // interleaved with src/sink relays. Each round is O(K); the loop exits
    // once the distance vector stops changing, typically after 2–3 rounds.
    // The residual graph of an optimal MCF has no negative cycles, so the
    // fixpoint is well-defined; the loop is capped at K+2 rounds as a safety
    // net (matching the Bellman-Ford bound) but real inputs exit much sooner.
    //
    // Requires: at least one ChainEdge present (the caller is responsible).
    // Fills _chain_R_buf, _chain_L_buf, and all flag/cost scratch arrays from
    // the current solved flow. Must be called before _chain_run_search().
    void _chain_build_search_state() const {
        const auto& topo = *_chain_topo;
        const size_t K = topo.order.size();
        const VALUE_TYPE INF = std::numeric_limits<VALUE_TYPE>::max();

        auto& c_right = _chain_R_buf;
        auto& c_left  = _chain_L_buf;
        for (size_t g = 0; g + 1 < K; ++g) {
            const VALUE_TYPE R = _solver_flow(lemon_graph.arcFromId(topo.right_arc_ids[g]));
            const VALUE_TYPE L = _solver_flow(lemon_graph.arcFromId(topo.left_arc_ids[g]));
            c_right[g] = (L > 0) ? -topo.gap_cost[g] : topo.gap_cost[g];
            c_left[g]  = (R > 0) ? -topo.gap_cost[g] : topo.gap_cost[g];
        }

        auto& has_src_fwd    = _chain_has_src_fwd;   auto& has_src_rev    = _chain_has_src_rev;
        auto& has_sink_fwd   = _chain_has_sink_fwd;  auto& has_sink_rev   = _chain_has_sink_rev;
        auto& exp_trash_cost = _chain_exp_trash_cost; auto& theo_trash_cost = _chain_theo_trash_cost;
        auto& exp_trash_fwd  = _chain_exp_trash_fwd;  auto& exp_trash_rev  = _chain_exp_trash_rev;
        auto& theo_trash_fwd = _chain_theo_trash_fwd; auto& theo_trash_rev = _chain_theo_trash_rev;
        std::fill(has_src_fwd.begin(),     has_src_fwd.end(),     uint8_t(0));
        std::fill(has_src_rev.begin(),     has_src_rev.end(),     uint8_t(0));
        std::fill(has_sink_fwd.begin(),    has_sink_fwd.end(),    uint8_t(0));
        std::fill(has_sink_rev.begin(),    has_sink_rev.end(),    uint8_t(0));
        std::fill(exp_trash_cost.begin(),  exp_trash_cost.end(),  INF);
        std::fill(theo_trash_cost.begin(), theo_trash_cost.end(), INF);
        std::fill(exp_trash_fwd.begin(),   exp_trash_fwd.end(),   uint8_t(0));
        std::fill(exp_trash_rev.begin(),   exp_trash_rev.end(),   uint8_t(0));
        std::fill(theo_trash_fwd.begin(),  theo_trash_fwd.end(),  uint8_t(0));
        std::fill(theo_trash_rev.begin(),  theo_trash_rev.end(),  uint8_t(0));
        for (LEMON_INDEX ii = 0; ii < static_cast<LEMON_INT>(edges.size()); ++ii) {
            const auto& et = edges[ii].get_type();
            if (std::holds_alternative<SrcToEmpiricalEdge>(et)) {
                const size_t pos = topo.node_to_pos[edges[ii].get_end_node_id()];
                if (pos == std::numeric_limits<size_t>::max()) continue;
                auto arc = lemon_graph.arcFromId(ii);
                VALUE_TYPE flow = _solver_flow(arc), cap = capacities_map[arc];
                if (flow < cap) has_src_fwd[pos] = true;
                if (flow > 0) has_src_rev[pos] = true;
            } else if (std::holds_alternative<TheoreticalToSinkEdge>(et)) {
                const size_t pos = topo.node_to_pos[edges[ii].get_start_node_id()];
                if (pos == std::numeric_limits<size_t>::max()) continue;
                auto arc = lemon_graph.arcFromId(ii);
                VALUE_TYPE flow = _solver_flow(arc), cap = capacities_map[arc];
                if (flow < cap) has_sink_fwd[pos] = true;
                if (flow > 0) has_sink_rev[pos] = true;
            } else if (const auto* e = std::get_if<EmpiricalTrashEdge>(&et)) {
                const size_t pos = topo.node_to_pos[edges[ii].get_start_node_id()];
                if (pos == std::numeric_limits<size_t>::max()) continue;
                auto arc = lemon_graph.arcFromId(ii);
                VALUE_TYPE flow = _solver_flow(arc), cap = capacities_map[arc];
                exp_trash_cost[pos] = e->get_cost();
                if (flow < cap) exp_trash_fwd[pos] = true;
                if (flow > 0)   exp_trash_rev[pos] = true;
            } else if (const auto* t = std::get_if<TheoreticalTrashEdge>(&et)) {
                const size_t pos = topo.node_to_pos[edges[ii].get_end_node_id()];
                if (pos == std::numeric_limits<size_t>::max()) continue;
                auto arc = lemon_graph.arcFromId(ii);
                VALUE_TYPE flow = _solver_flow(arc), cap = capacities_map[arc];
                theo_trash_cost[pos] = t->get_cost();
                if (flow < cap) theo_trash_fwd[pos] = true;
                if (flow > 0)   theo_trash_rev[pos] = true;
            }
        }
    }

    // Runs the relay+sweep search from source_id using pre-filled scratch state.
    // _chain_build_search_state() must have been called first.
    std::vector<VALUE_TYPE> _chain_run_search(LEMON_INDEX source_id) const {
        const auto& topo = *_chain_topo;
        const size_t K = topo.order.size();
        const VALUE_TYPE INF = std::numeric_limits<VALUE_TYPE>::max();
        const LEMON_INDEX src_id = 0;
        const LEMON_INDEX sink_id = 1;

        auto& dist         = _chain_dist_buf;
        auto& c_right      = _chain_R_buf;
        auto& c_left       = _chain_L_buf;
        auto& has_src_fwd  = _chain_has_src_fwd;   auto& has_src_rev  = _chain_has_src_rev;
        auto& has_sink_fwd = _chain_has_sink_fwd;  auto& has_sink_rev = _chain_has_sink_rev;
        auto& exp_trash_cost  = _chain_exp_trash_cost; auto& theo_trash_cost = _chain_theo_trash_cost;
        auto& exp_trash_fwd   = _chain_exp_trash_fwd;  auto& exp_trash_rev  = _chain_exp_trash_rev;
        auto& theo_trash_fwd  = _chain_theo_trash_fwd; auto& theo_trash_rev = _chain_theo_trash_rev;

        std::fill(dist.begin(), dist.end(), INF);
        dist[source_id] = 0;

        bool changed = false;
        auto update_min = [&](VALUE_TYPE& a, VALUE_TYPE b) { if (b < a) { a = b; changed = true; } };

        auto relay = [&]() {
            for (size_t i = 0; i < K; ++i) {
                const VALUE_TYPE d = dist[topo.order[i]];
                if (d == INF) continue;
                // Cost-0: reverse SrcToEmpiricalEdge, forward TheoreticalToSinkEdge.
                if (has_src_rev[i])  update_min(dist[src_id],  d);
                if (has_sink_fwd[i]) update_min(dist[sink_id], d);
                // Forward EmpiricalTrashEdge (Emp→Sink, cost +C_exp).
                if (exp_trash_fwd[i]) update_min(dist[sink_id], d + exp_trash_cost[i]);
                // Reverse TheoreticalTrashEdge (Theo→Source, cost -C_theo).
                if (theo_trash_rev[i]) update_min(dist[src_id], d - theo_trash_cost[i]);
            }
            const VALUE_TYPE ds = dist[src_id], dk = dist[sink_id];
            for (size_t i = 0; i < K; ++i) {
                // Cost-0: forward SrcToEmpiricalEdge, reverse TheoreticalToSinkEdge.
                if (ds != INF && has_src_fwd[i])  update_min(dist[topo.order[i]], ds);
                if (dk != INF && has_sink_rev[i]) update_min(dist[topo.order[i]], dk);
                // Forward TheoreticalTrashEdge (Source→Theo, cost +C_theo).
                if (ds != INF && theo_trash_fwd[i]) update_min(dist[topo.order[i]], ds + theo_trash_cost[i]);
                // Reverse EmpiricalTrashEdge (Sink→Emp, cost -C_exp).
                if (dk != INF && exp_trash_rev[i]) update_min(dist[topo.order[i]], dk - exp_trash_cost[i]);
            }
        };
        auto chain_sweep = [&]() {
            for (size_t i = 1; i < K; ++i) {
                const VALUE_TYPE d = dist[topo.order[i-1]];
                if (d != INF) update_min(dist[topo.order[i]], d + c_right[i-1]);
            }
            for (size_t ii = 1; ii < K; ++ii) {
                const size_t i = K - 1 - ii;
                const VALUE_TYPE d = dist[topo.order[i+1]];
                if (d != INF) update_min(dist[topo.order[i]], d + c_left[i]);
            }
        };

        const size_t MAX_ROUNDS = K + 2;
        for (size_t round = 0; round < MAX_ROUNDS; ++round) {
            changed = false;
            relay();
            chain_sweep();
            if (!changed) break;
        }
        return dist;
    }

    std::vector<VALUE_TYPE> chain_residual_distances(LEMON_INDEX source_id) const {
        if (!_chain_topo.has_value())
            throw std::runtime_error(
                "chain_residual_distances() called on non-chain subgraph.");
        _chain_build_search_state();
        return _chain_run_search(source_id);
    }

    bool has_chain_edges() const {
        return _chain_topo.has_value();
    }

    // Returns the node IDs of the chain in sorted position order, or an empty
    // vector if this subgraph has no chain topology.  Used by update_positions_and_solve
    // to validate that new positions don't change the chain's sorted order.
    const std::vector<LEMON_INDEX>& get_chain_order() const {
        static const std::vector<LEMON_INDEX> empty;
        return _chain_topo.has_value() ? _chain_topo->order : empty;
    }

    // Accumulate position gradients from this subgraph into caller-owned spans.
    // emp_grad is flat [N_emp * DIM] row-major; theo_grads[s] is [N_s * DIM] row-major.
    // Caller must zero both before the first call (multiple subgraphs accumulate additively).
    // Only MatchingEdge arcs with nonzero flow contribute.  Allocates nothing.
    template<typename Distribution_t, typename DistMetric>
    void accumulate_position_gradients(
        const Distribution_t* new_empirical,
        const std::vector<Distribution_t*>& new_theoretical,
        std::span<double> emp_grad,
        std::vector<std::span<double>>& theo_grads,
        double p = 1.0
    ) const {
        static constexpr size_t DIM = std::tuple_size_v<typename Distribution_t::Point_t>;
        for (const auto& e : _matching_edge_cache) {
            const VALUE_TYPE flow = _solver_flow(e.arc);
            if (flow == 0) continue;
            const auto emp_pt  = new_empirical->get_point(e.emp_peak_index);
            const auto theo_pt = new_theoretical[e.spectrum_id]->get_point(e.theo_peak_index);
            const auto g = DistMetric::grad_x(emp_pt, theo_pt);
            // Cost is d^p, so d(cost)/dx = p * d^(p-1) * grad_x(d).  p == 1 gives
            // factor 1 (bit-identical to the legacy W_1 gradient).  The gradient is
            // in REAL units (independent of the cost scale).
            double factor = 1.0;
            if (p != 1.0) {
                const double d = DistMetric::dist(emp_pt, theo_pt);
                factor = p * std::pow(d, p - 1.0);
            }
            for (size_t d = 0; d < DIM; ++d) {
                emp_grad[e.emp_peak_index * DIM + d]                   += static_cast<double>(flow) * factor * g[d];
                theo_grads[e.spectrum_id][e.theo_peak_index * DIM + d] -= static_cast<double>(flow) * factor * g[d];
            }
        }
    }

    // Accumulate position gradients for a chain (1D) subgraph.
    // Total cost = sum_g (R[g]+L[g])*gap_g.  Moving a peak at chain position k
    // changes gap_{k-1} by +delta and gap_k by -delta, so the gradient is
    // dir_sign*(left_total - right_total).  dir_sign is +1 for an ascending
    // chain (pos[0] < pos[1]) and -1 for descending.
    // DistMetric is accepted for API symmetry but unused (all 1D metrics agree).
    template<typename Distribution_t, typename DistMetric>
    void accumulate_position_gradients_chain(
        const Distribution_t* new_empirical,
        const std::vector<Distribution_t*>& new_theoretical,
        std::span<double> emp_grad,
        std::vector<std::span<double>>& theo_grads
    ) const {
        static constexpr size_t DIM = std::tuple_size_v<typename Distribution_t::Point_t>;
        static_assert(DIM == 1,
            "accumulate_position_gradients_chain requires 1D distributions");

        if (!_chain_topo.has_value()) return;
        const auto& topo = *_chain_topo;
        const size_t K = topo.order.size();
        if (K < 2) return;

        auto& R = _chain_R_buf;
        auto& L = _chain_L_buf;
        for (size_t g = 0; g < K - 1; ++g) {
            R[g] = _solver_flow(lemon_graph.arcFromId(topo.right_arc_ids[g]));
            L[g] = _solver_flow(lemon_graph.arcFromId(topo.left_arc_ids[g]));
        }

        auto get_node_pos = [&](LEMON_INDEX nid) -> double {
            const auto& ntype = nodes[nid].get_type();
            if (const auto* emp = std::get_if<EmpiricalNode<intensity_type>>(&ntype))
                return new_empirical->get_point(emp->get_peak_index())[0];
            if (const auto* theo = std::get_if<TheoreticalNode<intensity_type>>(&ntype))
                return new_theoretical[theo->get_spectrum_id()]->get_point(theo->get_peak_index())[0];
            return 0.0;
        };
        const double dir_sign =
            (get_node_pos(topo.order[1]) > get_node_pos(topo.order[0])) ? 1.0 : -1.0;

        for (size_t k = 0; k < K; ++k) {
            const VALUE_TYPE left_total  = (k > 0)     ? R[k-1] + L[k-1] : 0;
            const VALUE_TYPE right_total = (k < K - 1) ? R[k]   + L[k]   : 0;
            const double grad_val =
                dir_sign * static_cast<double>(left_total - right_total);

            const LEMON_INDEX nid = topo.order[k];
            const auto& ntype = nodes[nid].get_type();
            if (const auto* emp = std::get_if<EmpiricalNode<intensity_type>>(&ntype)) {
                emp_grad[emp->get_peak_index()] += grad_val;
            } else if (const auto* theo = std::get_if<TheoreticalNode<intensity_type>>(&ntype)) {
                theo_grads[theo->get_spectrum_id()][theo->get_peak_index()] += grad_val;
            }
        }
    }

    // Update MatchingEdge and ChainEdge costs in the already-built LEMON graph,
    // then immediately re-run the solver (warm-restarting for NetworkSimplex).
    // new_costs_per_edge_idx[i] is the new cost for edge i; entries for other
    // edge types (SrcToEmpirical, TheoreticalToSink, trash, ...) are ignored.
    // Precondition: build() and at least one solve() must have been called.
    void apply_new_costs(const std::vector<VALUE_TYPE>& new_costs_per_edge_idx) {
        if (!built)
            throw std::runtime_error("apply_new_costs() must be called after build().");
        if (new_costs_per_edge_idx.size() != edges.size())
            throw std::runtime_error("apply_new_costs(): cost vector size mismatch.");
        for (LEMON_INDEX ii = 0; ii < static_cast<LEMON_INT>(edges.size()); ++ii) {
            const auto& etype = edges[ii].get_type();
            if (std::holds_alternative<MatchingEdge>(etype) || std::holds_alternative<ChainEdge>(etype))
                costs_map[lemon_graph.arcFromId(ii)] = new_costs_per_edge_idx[ii];
        }
        // Re-sync gap_cost from costs_map so chain_residual_distances stays correct.
        if (_chain_topo.has_value()) {
            for (size_t g = 0; g < _chain_topo->right_arc_ids.size(); ++g)
                _chain_topo->gap_cost[g] = costs_map[lemon_graph.arcFromId(_chain_topo->right_arc_ids[g])];
        }
        _run_solver(/*costs_changed=*/true);
    }

    // Dijkstra variant of residual shortest-path using NetworkSimplex potentials.
    // After an optimal NS solve, reduced costs c_r(u,v) = c(u,v) + pi[u] - pi[v]
    // are >= 0 on every residual arc (complementary slackness), so Dijkstra applies.
    // O((V+E) log V) vs Bellman-Ford's O(V*E).
    // true_dist[v] = dijkstra_reduced_dist[v] + pi[v] - pi[source]
    std::vector<VALUE_TYPE> dijkstra_residual(LEMON_INDEX source_id) const {
        const LEMON_INDEX n = lemon_graph.nodeNum();
        const VALUE_TYPE INF = std::numeric_limits<VALUE_TYPE>::max();

        std::vector<VALUE_TYPE> pi(n);
        for (LEMON_INDEX i = 0; i < n; ++i)
            pi[i] = _solver_potential(lemon_graph.nodeFromId(i));

        std::vector<VALUE_TYPE> rdist(n, INF);
        rdist[source_id] = 0;
        using Entry = std::pair<VALUE_TYPE, LEMON_INDEX>;
        std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> pq;
        pq.emplace(0, source_id);

        while (!pq.empty()) {
            auto [d, u] = pq.top(); pq.pop();
            if (d > rdist[u]) continue;
            const auto u_node = lemon_graph.nodeFromId(u);

            // Forward residual: outgoing arcs of u
            for (lemon::StaticDigraph::OutArcIt a(lemon_graph, u_node); a != lemon::INVALID; ++a) {
                const LEMON_INDEX ii = lemon_graph.id(a);
                if (ii == simple_trash_idx) continue;
                if (!_unlimited_arc[ii] && _solver_flow(a) >= capacities_map[a]) continue;
                const LEMON_INDEX v = lemon_graph.id(lemon_graph.target(a));
                const VALUE_TYPE rcost = std::max<VALUE_TYPE>(0, costs_map[a] + pi[u] - pi[v]);
                const VALUE_TYPE nd = d + rcost;
                if (nd < rdist[v]) { rdist[v] = nd; pq.emplace(nd, v); }
            }

            // Reverse residual: incoming arcs of u (arc goes v→u in original, u→v in residual)
            for (lemon::StaticDigraph::InArcIt a(lemon_graph, u_node); a != lemon::INVALID; ++a) {
                const LEMON_INDEX ii = lemon_graph.id(a);
                if (ii == simple_trash_idx) continue;
                if (_solver_flow(a) <= 0) continue;
                const LEMON_INDEX v = lemon_graph.id(lemon_graph.source(a));
                const VALUE_TYPE rcost = std::max<VALUE_TYPE>(0, -costs_map[a] + pi[u] - pi[v]);
                const VALUE_TYPE nd = d + rcost;
                if (nd < rdist[v]) { rdist[v] = nd; pq.emplace(nd, v); }
            }
        }

        // Convert reduced distances back to true distances
        std::vector<VALUE_TYPE> dist(n, INF);
        const VALUE_TYPE pi_src = pi[source_id];
        for (LEMON_INDEX i = 0; i < n; ++i)
            if (rdist[i] != INF)
                dist[i] = rdist[i] + pi[i] - pi_src;
        return dist;
    }

    // Compute single-source shortest distances on the implicit residual graph
    // (excluding the trash edge). For each arc:
    //   forward residual if flow < capacity (cost = original)
    //   reverse residual if flow > 0 (cost = -original)
    std::vector<VALUE_TYPE> bellman_ford_residual(LEMON_INDEX source_id) const {
        const LEMON_INDEX n = lemon_graph.nodeNum();
        const LEMON_INDEX m = lemon_graph.arcNum();
        const VALUE_TYPE INF = std::numeric_limits<VALUE_TYPE>::max();
        std::vector<VALUE_TYPE> dist(n, INF);
        dist[source_id] = 0;

        for (LEMON_INDEX iter = 0; iter < n - 1; ++iter) {
            bool changed = false;
            for (LEMON_INDEX ii = 0; ii < m; ++ii) {
                if (ii == simple_trash_idx) continue;
                auto arc = lemon_graph.arcFromId(ii);
                LEMON_INDEX u = lemon_graph.id(lemon_graph.source(arc));
                LEMON_INDEX v = lemon_graph.id(lemon_graph.target(arc));
                VALUE_TYPE cost = costs_map[arc];
                VALUE_TYPE cap = capacities_map[arc];
                VALUE_TYPE flow = _solver_flow(arc);
                // Matching edges have unlimited base capacity; the LEMON
                // capacity (min of endpoint intensities) is an optimization
                // that should not limit the residual graph. Chain edges
                // also represent unlimited-capacity residual transitions.
                bool unlimited = std::holds_alternative<MatchingEdge>(edges[ii].get_type())
                              || std::holds_alternative<ChainEdge>(edges[ii].get_type());

                if ((unlimited || flow < cap) && dist[u] != INF && dist[u] + cost < dist[v]) {
                    dist[v] = dist[u] + cost;
                    changed = true;
                }
                if (flow > 0 && dist[v] != INF && dist[v] - cost < dist[u]) {
                    dist[u] = dist[v] - cost;
                    changed = true;
                }
            }
            if (!changed) break;
        }
        return dist;
    }

    // Opt-in (GradientMode::DualPi) fast, approximate replacement for
    // dijkstra_residual(): the pure dual-potential difference
    //   dist[i] = pi[i] - pi[source_id].
    // This is exactly the residual-shortest-path distance with the (>= 0)
    // reduced-cost detour term dropped, so it is correct only for nodes on
    // the optimal flow support (reduced-cost-0 reachable) and a lower bound
    // elsewhere; it is also basis-dependent at degenerate optima.  O(n), no
    // search.  NOT value-equivalent to the residual marginal — only reachable
    // via the *_fast_approx() entry points.
    std::vector<VALUE_TYPE> pi_distances(LEMON_INDEX source_id) const {
        const LEMON_INDEX n = lemon_graph.nodeNum();
        std::vector<VALUE_TYPE> dist(n);
        const VALUE_TYPE pi_src =
            _solver_potential(lemon_graph.nodeFromId(source_id));
        for (LEMON_INDEX i = 0; i < n; ++i)
            dist[i] = _solver_potential(lemon_graph.nodeFromId(i)) - pi_src;
        return dist;
    }

    // Per-peak marginal cost of increasing each theoretical signal by 1.
    // Returns vector of (spectrum_id, peak_index, derivative).
    //
    // Simple trash: Bellman-Ford excludes the trash edge; src_adjust/sink_adjust
    // manually account for the Source↔Sink shortcut it provides.
    //
    // Asymmetric trash: full residual already contains the shortcuts (reverse
    // EmpiricalTrashEdge: Sink→Emp at -C_exp; forward TheoreticalTrashEdge:
    // Source→Theo at C_theo), so no adjustments are needed.
    //   E > T (supply fixed): augmenting cycle through T_node uses dist_sink.
    //   T >= E (supply +1):   extra Source unit routed to T_node uses dist_src.
    // (DerivContext is declared near the top of the class so it can be cached.)

    // want_pi=false: exact residual marginals.  want_pi=true: fast dual-pi
    // approximation (only when the NetworkSimplex solver is in use; chain /
    // non-NS subgraphs have no potentials so they ignore want_pi and stay
    // exact).
    DerivContext _make_deriv_context(bool want_pi) const {
        if (!_solver_has_value())
            throw std::runtime_error("Must call solve() before signal_part_derivatives().");
        if (simple_trash_idx == std::numeric_limits<LEMON_INDEX>::max()
                && !experimental_trash_added && !theoretical_trash_added)
            throw std::runtime_error("signal_part_derivatives() requires trash edges.");

        DerivContext ctx;
        ctx.INF = std::numeric_limits<VALUE_TYPE>::max();
        ctx.supply_fixed = (lemon_empirical_intensity > lemon_theoretical_intensity);
        ctx.asymmetric = experimental_trash_added || theoretical_trash_added;
        const LEMON_INDEX sink_id = 1;
        const bool use_chain = has_chain_edges();
        const bool use_dijkstra = !use_chain &&
            (_use_lct() ? ns_lct_solver.has_value() : ns_solver.has_value());
        const bool need_src  = !ctx.asymmetric || !ctx.supply_fixed;
        const bool need_sink = !ctx.asymmetric ||  ctx.supply_fixed;
        if (use_chain) {
            _chain_build_search_state();
            if (need_src)  ctx.dist_src  = _chain_run_search(0);
            if (need_sink) ctx.dist_sink = _chain_run_search(sink_id);
        } else {
            const bool use_pi = want_pi && use_dijkstra;
            auto compute_dist = [&](LEMON_INDEX id) -> std::vector<VALUE_TYPE> {
                if (use_pi)       return pi_distances(id);
                return use_dijkstra ? dijkstra_residual(id)
                                    : bellman_ford_residual(id);
            };
            if (need_src)  ctx.dist_src  = compute_dist(0);
            if (need_sink) ctx.dist_sink = compute_dist(sink_id);
        }

        ctx.trash_cost = 0; ctx.src_adjust = 0; ctx.sink_adjust = 0;
        if (!ctx.asymmetric) {
            // Scaled trash cost (same units as the residual distances, which
            // come from the scaled costs_map).
            ctx.trash_cost  = costs_map[lemon_graph.arcFromId(simple_trash_idx)];
            ctx.src_adjust  = ctx.supply_fixed ? -ctx.trash_cost : 0;
            ctx.sink_adjust = ctx.supply_fixed ?  0 : ctx.trash_cost;
        }

        ctx.theo_sink_slack.assign(nodes.size(), VALUE_TYPE(-1));
        for (const auto& e : _theo_sink_edge_cache) {
            const LEMON_INDEX node_id = lemon_graph.id(lemon_graph.source(e.arc));
            ctx.theo_sink_slack[node_id] = capacities_map[e.arc] - _solver_flow(e.arc);
        }
        return ctx;
    }

    // Value-exact memoized accessor: rebuilds the context only when the
    // solution changed since the last build (_solution_version), otherwise
    // returns the cached one.  Eliminates the redundant residual recompute
    // when several derivative queries (spectrum + signal + repeated/identical
    // re-solves) hit the same solved solution.  Bit-identical to
    // _make_deriv_context() because the context is a pure function of the
    // post-solve solver state.
    const DerivContext& _get_deriv_context(bool use_pi) const {
        const int k = use_pi ? 1 : 0;
        if (_deriv_ctx_version[k] != _solution_version) {
            _deriv_ctx_cache[k] = _make_deriv_context(use_pi);
            _deriv_ctx_version[k] = _solution_version;
        }
        return _deriv_ctx_cache[k];
    }

    VALUE_TYPE _node_deriv(LEMON_INDEX node_id, const DerivContext& ctx) const {
        const VALUE_TYPE slack = ctx.theo_sink_slack[node_id];
        if (slack != VALUE_TYPE(-1) && slack > 0 && ctx.supply_fixed)
            return VALUE_TYPE(0);
        VALUE_TYPE deriv;
        if (ctx.asymmetric) {
            deriv = ctx.supply_fixed ? ctx.dist_sink[node_id] : ctx.dist_src[node_id];
            if (deriv == ctx.INF) deriv = 0;
        } else {
            deriv = ctx.trash_cost;
            if (!ctx.dist_src.empty() && ctx.dist_src[node_id] != ctx.INF)
                deriv = std::min(deriv, ctx.dist_src[node_id] + ctx.src_adjust);
            if (!ctx.dist_sink.empty() && ctx.dist_sink[node_id] != ctx.INF)
                deriv = std::min(deriv, ctx.dist_sink[node_id] + ctx.sink_adjust);
        }
        return deriv;
    }

    std::vector<std::tuple<size_t, LEMON_INDEX, VALUE_TYPE>>
    _signal_part_derivatives_impl(const DerivContext& ctx) const {
        std::vector<std::tuple<size_t, LEMON_INDEX, VALUE_TYPE>> result;
        for (const auto& node : nodes) {
            auto* theo = std::get_if<TheoreticalNode<intensity_type>>(&node.get_type());
            if (!theo) continue;
            result.emplace_back(theo->get_spectrum_id(), theo->get_peak_index(),
                                _node_deriv(node.get_id(), ctx));
        }
        return result;
    }

    std::vector<std::pair<size_t, double>>
    _spectrum_proportion_derivatives_impl(const DerivContext& ctx) const {
        // Weight each peak's integer per-supply marginal by its REAL (double)
        // intensity — d(cost)/dw = sum_i marginal_i * real_intensity_i.  The
        // intensity scale is deliberately absent (it cancels: it enters both
        // dS/dw and the total_cost unscale), and the real intensity may be
        // fractional, so this must accumulate in double, not the integer
        // VALUE_TYPE (which would floor sub-unit intensities to zero).
        std::vector<double> accum(no_target_distributions, 0.0);
        for (const auto& node : nodes) {
            auto* theo = std::get_if<TheoreticalNode<intensity_type>>(&node.get_type());
            if (!theo) continue;
            accum[theo->get_spectrum_id()] +=
                static_cast<double>(_node_deriv(node.get_id(), ctx))
                * static_cast<double>(theo->get_intensity());
        }
        std::vector<std::pair<size_t, double>> result;
        result.reserve(no_target_distributions);
        for (size_t s = 0; s < no_target_distributions; ++s)
            result.emplace_back(s, accum[s]);
        return result;
    }

public:
    // Per-peak marginal cost of increasing each theoretical signal by 1
    // (spectrum_id, peak_index, derivative).  Exact: cheapest residual
    // augmenting route (Dijkstra on reduced costs).
    std::vector<std::tuple<size_t, LEMON_INDEX, VALUE_TYPE>> signal_part_derivatives() const {
        return _signal_part_derivatives_impl(_get_deriv_context(/*use_pi=*/false));
    }

    // Fast, APPROXIMATE per-peak marginals: the pure dual-potential
    // difference (pi[v] - pi[src]), skipping the residual search.  A lower
    // bound on the true marginal — exact only for peaks on the optimal flow
    // support, basis-dependent at degenerate optima.  Different VALUES from
    // signal_part_derivatives(); opt-in only.
    std::vector<std::tuple<size_t, LEMON_INDEX, VALUE_TYPE>>
    signal_part_derivatives_fast_approx() const {
        return _signal_part_derivatives_impl(_get_deriv_context(/*use_pi=*/true));
    }

    // Gradient of total cost w.r.t. scaling each spectrum's proportion
    // (spectrum_id, derivative) = sum_i(peak_derivative_i * intensity_i).
    // Exact residual marginals.
    std::vector<std::pair<size_t, double>> spectrum_proportion_derivatives() const {
        return _spectrum_proportion_derivatives_impl(_get_deriv_context(/*use_pi=*/false));
    }

    // Fast, APPROXIMATE spectrum-proportion gradient (dual-potential
    // difference; see signal_part_derivatives_fast_approx for the accuracy
    // caveat).  Different VALUES from spectrum_proportion_derivatives().
    std::vector<std::pair<size_t, double>>
    spectrum_proportion_derivatives_fast_approx() const {
        return _spectrum_proportion_derivatives_impl(_get_deriv_context(/*use_pi=*/true));
    }
};

template <typename VALUE_TYPE, typename intensity_type>
class WassersteinNetwork {
    std::vector<FlowNode<intensity_type>> nodes;
    std::vector<FlowEdge<intensity_type>> edges;

    const size_t _no_theoretical_spectra;
    const std::vector<size_t> _theoretical_spectra_sizes;

    std::vector<LEMON_INDEX> dead_end_node_ids;
    std::vector<std::unique_ptr<WassersteinNetworkSubgraph<VALUE_TYPE, intensity_type>>> flow_subgraphs;

    intensity_type _isolated_empirical_intensity = 0;
    std::vector<intensity_type> _isolated_theoretical_intensity;
    // Real (unscaled) per-unit trash costs for isolated/dead-end nodes.
    double _isolated_exp_trash_cost = 0;
    double _isolated_theo_trash_cost = 0;
    std::vector<double> _last_point;

    bool built = false;

    // Wasserstein transport order p.  Edge cost = ground_distance^p, so the
    // network optimises/reports the W_p^p objective; the ^(1/p) root is applied
    // by the high-level Python wrappers.  p == 1 reproduces the legacy behaviour
    // bit-for-bit.  p != 1 requires the dense factory (chain cost is not additive
    // under exponentiation).
    double _p_order = 1.0;

    // Cost scaling: integer edge costs are quantize_cost(real, _scale, p==1).
    // _scale is chosen at build() from the largest real cost (matching + trash).
    // p == 1 forces _scale == 1 (legacy truncation).  _max_real_cost tracks the
    // running maximum real edge/trash cost so build() can size _scale.
    int64_t _scale = 1;
    double _max_real_cost = 0.0;

    // Intensity scaling: real (double) node intensities map to integer LEMON
    // supplies as round-toward-zero(real * _intensity_scale).  Set via
    // set_intensity_scale() before build(); propagated to every subgraph in
    // build().  1.0 (the default) reproduces the legacy verbatim-intensity
    // behaviour.  total_cost() and the proportion derivatives are in scaled
    // units (cost_scale * intensity_scale); the Python wrapper unscales by
    // scale_factor() * intensity_scale_factor().
    double _intensity_scale = 1.0;

public:
    WassersteinNetwork(std::vector<FlowNode<intensity_type>>&& nodes_,
                       std::vector<FlowEdge<intensity_type>>&& edges_,
                       size_t no_theoretical_spectra_,
                       std::vector<size_t>&& theoretical_spectra_sizes_,
                       std::vector<LEMON_INDEX>&& dead_end_node_ids_,
                       double p_order = 1.0,
                       double max_real_cost = 0.0
    ) :
    nodes(std::move(nodes_)),
    edges(std::move(edges_)),
    _no_theoretical_spectra(no_theoretical_spectra_),
    _theoretical_spectra_sizes(std::move(theoretical_spectra_sizes_)),
    dead_end_node_ids(std::move(dead_end_node_ids_)),
    _p_order(p_order),
    _max_real_cost(max_real_cost)
    {
        build_subgraphs();
    };

    double p_order() const { return _p_order; }
    int64_t scale_factor() const { return _scale; }
    double intensity_scale_factor() const { return _intensity_scale; }
    // Must be called before build() to take effect (build() propagates it to
    // the subgraphs and folds it into the cost-scale overflow budget).
    // The integer-intensity backend does no scaling at all: intensities are
    // already exact integers, so the only legal scale is 1.
    void set_intensity_scale(double s) {
        if constexpr (std::is_integral_v<intensity_type>) {
            if (s != 1.0)
                throw std::invalid_argument(
                    "Integer intensity backend does not support intensity scaling "
                    "(set_intensity_scale requires 1; use the double-intensity backend).");
            _intensity_scale = 1.0;
        } else {
            _intensity_scale = s;
        }
    }

    WassersteinNetwork(const WassersteinNetwork&) = delete;
    WassersteinNetwork& operator=(const WassersteinNetwork&) = delete;
    WassersteinNetwork(WassersteinNetwork&& other) :
        nodes(std::move(other.nodes)),
        edges(std::move(other.edges)),
        _no_theoretical_spectra(other._no_theoretical_spectra),
        _theoretical_spectra_sizes(std::move(other._theoretical_spectra_sizes)),
        dead_end_node_ids(std::move(other.dead_end_node_ids)),
        flow_subgraphs(std::move(other.flow_subgraphs)),
        _isolated_empirical_intensity(other._isolated_empirical_intensity),
        _isolated_theoretical_intensity(std::move(other._isolated_theoretical_intensity)),
        _isolated_exp_trash_cost(other._isolated_exp_trash_cost),
        _isolated_theo_trash_cost(other._isolated_theo_trash_cost),
        _last_point(std::move(other._last_point)),
        built(other.built),
        _p_order(other._p_order),
        _scale(other._scale),
        _max_real_cost(other._max_real_cost)
    {
        other.built = false;
    }
    WassersteinNetwork& operator=(WassersteinNetwork&& other) = delete;
    size_t no_nodes() const {
        return nodes.size();
    };
    size_t no_edges() const {
        return edges.size();
    };
    size_t no_theoretical_spectra() const {
        return _no_theoretical_spectra;
    };

    const std::vector<size_t>& theoretical_spectra_sizes() const {
        return _theoretical_spectra_sizes;
    };

    const std::vector<FlowNode<intensity_type>>& get_nodes() const {
        return nodes;
    };
    const std::vector<FlowEdge<intensity_type>>& get_edges() const {
        return edges;
    };

    std::vector<std::vector<LEMON_INDEX>> neighbourhood_lists() const {
        std::vector<std::vector<LEMON_INDEX>> neighbourhood_lists;
        neighbourhood_lists.resize(nodes.size());
        for (const auto& edge : edges) {
            const LEMON_INDEX start_node_id = edge.get_start_node_id();
            const LEMON_INDEX end_node_id = edge.get_end_node_id();
            neighbourhood_lists[start_node_id].push_back(end_node_id);
            neighbourhood_lists[end_node_id].push_back(start_node_id);
        }
        return neighbourhood_lists;
    };

    std::pair<std::vector<std::vector<LEMON_INDEX>>, std::vector<LEMON_INDEX>> split_into_subgraphs() const {
        std::vector<std::vector<LEMON_INDEX>> subgraphs;
        std::vector<LEMON_INDEX> dead_end_nodes;

        assert_fits_lemon_index(nodes.size(), "network nodes");
        std::vector<bool> visited(nodes.size(), false);
        visited[0] = true; // Mark the source node as visited
        visited[1] = true; // Mark the sink node as visited
        std::vector<LEMON_INDEX> stack;
        std::vector<std::vector<LEMON_INDEX>> neighbourhood_lists = this->neighbourhood_lists();

        for (LEMON_INDEX node_id = 0; node_id < static_cast<LEMON_INT>(nodes.size()); ++node_id) {
            if (!visited[node_id]) {
                std::vector<LEMON_INDEX>& neighbours = neighbourhood_lists[node_id];
                if(neighbours.size() == 0) {
                    dead_end_nodes.push_back(node_id);
                } else {
                    std::vector<LEMON_INDEX> subgraph;
                    stack.push_back(node_id);
                    while (!stack.empty()) {
                        LEMON_INDEX current_node = stack.back();
                        stack.pop_back();
                        if (!visited[current_node]) {
                            visited[current_node] = true;
                            subgraph.push_back(current_node);
                            for (LEMON_INDEX neighbour : neighbourhood_lists[current_node]) {
                                if (!visited[neighbour]) {
                                    stack.push_back(neighbour);
                                }
                            }
                        }
                    }
                    // TODO: potentially remove this
                    std::sort(subgraph.begin(), subgraph.end());
                    subgraphs.push_back(subgraph);
                }
            }
        }
        return {subgraphs, dead_end_nodes};
    }

    void build_subgraphs() {
        auto [_subgraphs, _dead_end_nodes] = this->split_into_subgraphs();

        dead_end_node_ids = std::move(_dead_end_nodes);

        std::unique_ptr<LEMON_INDEX[]> node_in_subgraph = std::make_unique<LEMON_INDEX[]>(nodes.size());

        #ifdef LEMON_DO_ASSERTS
        for (size_t ii = 0; ii < nodes.size(); ++ii)
            node_in_subgraph[ii] = -10;
        #endif

        for (LEMON_INDEX subgraph_idx = 0; subgraph_idx < static_cast<LEMON_INT>(_subgraphs.size()); ++subgraph_idx)
            for (const auto& node_id : _subgraphs[subgraph_idx])
                node_in_subgraph[node_id] = subgraph_idx;

        #ifdef WNET_DO_ASSERTS
        for(auto dead_end_node_id : dead_end_node_ids)
            node_in_subgraph[dead_end_node_id] = -1;
        for(size_t node_id = 0; node_id < nodes.size(); ++node_id)
            if(node_in_subgraph[node_id] == -10)
                throw std::runtime_error("Node not assigned to any subgraph");
        #endif

        std::vector<std::vector<FlowEdge<intensity_type>*>> subgraph_edges(_subgraphs.size());
        for (auto& edge : edges)
        {
            const LEMON_INDEX start_node_id = edge.get_start_node_id();
            const LEMON_INDEX start_subgraph_idx = node_in_subgraph[start_node_id];
            subgraph_edges[start_subgraph_idx].push_back(&edge);

            #ifdef WNET_DO_ASSERTS
            const LEMON_INDEX end_node_id = edge.get_end_node_id();
            const LEMON_INDEX end_subgraph_idx = node_in_subgraph[end_node_id];
            if(start_subgraph_idx != end_subgraph_idx || start_subgraph_idx == -1)
                throw std::runtime_error("Edge connects nodes from different subgraphs or dead end nodes.");
            #endif
        }


        // TODO: optimize, right now this is needlessly O(subgraphs.size() * edges.size()),
        // can be O(subgraphs.size() + edges.size())
        flow_subgraphs.reserve(_subgraphs.size());
        for (size_t subgraph_idx = 0; subgraph_idx < _subgraphs.size(); ++subgraph_idx)
        {
            #ifdef DO_TONS_OF_PRINTS
            std::cout << "Subgraph" << std::endl;
            #endif
            flow_subgraphs.emplace_back(std::make_unique<WassersteinNetworkSubgraph<VALUE_TYPE, intensity_type>>(
                    _subgraphs[subgraph_idx],
                    nodes,
                    subgraph_edges[subgraph_idx],
                    _no_theoretical_spectra
            ));
        }
        _isolated_theoretical_intensity.assign(_no_theoretical_spectra, 0);
        for (LEMON_INDEX dead_end_id : dead_end_node_ids) {
            std::visit([&](const auto& t) {
                using T = std::decay_t<decltype(t)>;
                if constexpr (std::is_same_v<T, EmpiricalNode<intensity_type>>)
                    _isolated_empirical_intensity += t.get_intensity();
                else if constexpr (std::is_same_v<T, TheoreticalNode<intensity_type>>)
                    _isolated_theoretical_intensity[t.get_spectrum_id()] += t.get_intensity();
            }, nodes[dead_end_id].get_type());
        }
    }

    void add_simple_trash(double cost) {
        if (built)
            throw std::runtime_error("add_simple_trash() must be called before build(), not after.");
        _isolated_exp_trash_cost = cost;
        _isolated_theo_trash_cost = cost;
        _max_real_cost = std::max(_max_real_cost, cost);
        for (auto& flow_subgraph : flow_subgraphs)
            flow_subgraph->add_simple_trash(cost);
    };

    void add_experimental_trash(double cost) {
        if (built)
            throw std::runtime_error("add_experimental_trash() must be called before build().");
        _isolated_exp_trash_cost = cost;
        _max_real_cost = std::max(_max_real_cost, cost);
        for (auto& flow_subgraph : flow_subgraphs)
            flow_subgraph->add_experimental_trash(cost);
    };

    void add_theoretical_trash(double cost) {
        if (built)
            throw std::runtime_error("add_theoretical_trash() must be called before build().");
        _isolated_theo_trash_cost = cost;
        _max_real_cost = std::max(_max_real_cost, cost);
        for (auto& flow_subgraph : flow_subgraphs)
            flow_subgraph->add_theoretical_trash(cost);
    };

    void build(SolverConfig config = NetworkSimplexConfig{}) {
        // Total flow upper bound for the cost-scale accumulator ceiling: the
        // per-subgraph flow is max(emp, theo) intensity, so summing all node
        // intensities over-estimates the network-wide flow (safe — it only
        // shrinks the scale).  Assumes solve()'s point ~ O(1); a point that
        // scales theoretical intensity far above 1 can still overflow.
        double total_flow = 0.0;
        for (const auto& node : nodes)
            std::visit([&](const auto& n) {
                using T = std::decay_t<decltype(n)>;
                if constexpr (std::is_same_v<T, EmpiricalNode<intensity_type>> ||
                              std::is_same_v<T, TheoreticalNode<intensity_type>>)
                    total_flow += static_cast<double>(n.get_intensity());
            }, node.get_type());
        // Choose one global cost scale from the largest real cost across the whole
        // network (matching + trash), so every subgraph's integer costs — and the
        // summed total_cost — share the same units.  p == 1 keeps _scale == 1.
        // The integer flow the accumulator actually sees is the real flow times
        // the intensity scale, so size the cost-scale ceiling against that.
        _scale = pick_cost_scale(_max_real_cost, total_flow * _intensity_scale, _p_order == 1.0);
        const bool p_is_one = (_p_order == 1.0);
        for (auto& flow_subgraph : flow_subgraphs) {
            flow_subgraph->set_cost_scaling(_scale, p_is_one, _intensity_scale);
            flow_subgraph->build(config);
        }
        built = true;
    };

    // Quantised (scaled) isolated trash costs, matching the subgraph cost map.
    VALUE_TYPE _isolated_exp_trash_cost_scaled() const {
        return quantize_cost<VALUE_TYPE>(_isolated_exp_trash_cost, _scale, _p_order == 1.0);
    }
    VALUE_TYPE _isolated_theo_trash_cost_scaled() const {
        return quantize_cost<VALUE_TYPE>(_isolated_theo_trash_cost, _scale, _p_order == 1.0);
    }

    void solve()
    {
        std::vector<double> point(_no_theoretical_spectra, 1.0);
        solve(point);
    };

    void solve(const std::vector<double>& point) {
        if(!built)
            throw std::runtime_error("You must call build() before calling solve().");

        _last_point = point;
        for (auto& flow_subgraph : flow_subgraphs)
            flow_subgraph->set_point(point);
    };

    // Total cost in SCALED units (sum of scaled per-subgraph costs plus scaled
    // isolated-trash contributions).  The Python wrapper divides by scale_factor()
    // to recover the real W_p**p value.
    VALUE_TYPE total_cost() const {
        VALUE_TYPE cost = 0;
        for (const auto& flow_subgraph : flow_subgraphs)
            cost += flow_subgraph->total_cost();
        cost += _isolated_exp_trash_cost_scaled() * _isolated_empirical_intensity * _intensity_scale;
        const VALUE_TYPE theo_trash_scaled = _isolated_theo_trash_cost_scaled();
        for (size_t s = 0; s < _no_theoretical_spectra; ++s)
            cost += static_cast<VALUE_TYPE>(theo_trash_scaled * _isolated_theoretical_intensity[s] * _last_point[s] * _intensity_scale);
        return cost;
    };

    int warm_start_count() const {
        int total = 0;
        for (const auto& sg : flow_subgraphs)
            total += sg->warm_start_count();
        return total;
    }
    int cold_start_count() const {
        int total = 0;
        for (const auto& sg : flow_subgraphs)
            total += sg->cold_start_count();
        return total;
    }
    int dual_repair_count() const {
        int total = 0;
        for (const auto& sg : flow_subgraphs)
            total += sg->dual_repair_count();
        return total;
    }
    int primal_repair_count() const {
        int total = 0;
        for (const auto& sg : flow_subgraphs)
            total += sg->primal_repair_count();
        return total;
    }

    size_t no_subgraphs() const {
        return flow_subgraphs.size();
    };

    const WassersteinNetworkSubgraph<VALUE_TYPE, intensity_type>& get_subgraph(size_t idx) const {
        if (idx >= flow_subgraphs.size())
            throw std::out_of_range("Subgraph index out of range");
        return *flow_subgraphs[idx];
    };

    std::string to_string() const {
        std::string result;
        for (const auto& flow_subgraph : flow_subgraphs)
            result += flow_subgraph->to_string();
        return result;
    };

    std::string lemon_to_string() const {
        std::string result;
        for (const auto& flow_subgraph : flow_subgraphs)
            result += flow_subgraph->lemon_to_string();
        return result;
    };

    std::tuple<std::vector<LEMON_INDEX>, std::vector<LEMON_INDEX>, std::vector<VALUE_TYPE>> flows_for_target(size_t target_id) const {
        std::vector<LEMON_INDEX> empirical_peak_indices;
        std::vector<LEMON_INDEX> theoretical_peak_indices;
        std::vector<VALUE_TYPE> flows;
        for (const auto& flow_subgraph : flow_subgraphs)
            flow_subgraph->flows_for_target(target_id, empirical_peak_indices, theoretical_peak_indices, flows);
        return {empirical_peak_indices, theoretical_peak_indices, flows};
    };

    size_t count_matching_edges() const {
        size_t result = 0;
        for (const auto& edge : edges)
            std::visit([&](const auto& arg) {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, MatchingEdge>) result++;
            },
            edge.get_type());
        return result;
    }

    template<typename T>
    size_t count_nodes_of_type() const {
        size_t result = 0;
        for (const auto& node : nodes)
            if(std::holds_alternative<T>(node.get_type()))
                result++;
        return result;
    }

    template<typename T>
    size_t count_edges_of_type() const {
        size_t result = 0;
        for (const auto& edge : edges)
            if(std::holds_alternative<T>(edge.get_type()))
                result++;
        return result;
    }

    double matching_density() const {
        const double nominator = count_edges_of_type<MatchingEdge>() + count_edges_of_type<ChainEdge>() / 2.0;
        double denominator = 0;
        for (const auto& flow_subgraph : flow_subgraphs)
            denominator += flow_subgraph->template count_nodes_of_type<EmpiricalNode<intensity_type>>() * flow_subgraph->template count_nodes_of_type<TheoreticalNode<intensity_type>>();
        if (denominator == 0) return std::numeric_limits<double>::quiet_NaN();
        return nominator / denominator;
    }

    std::vector<std::tuple<size_t, LEMON_INDEX, VALUE_TYPE>>
    _signal_part_derivatives(bool fast) const {
        std::vector<std::tuple<size_t, LEMON_INDEX, VALUE_TYPE>> result;
        for (const auto& sg : flow_subgraphs) {
            auto sg_derivs = fast ? sg->signal_part_derivatives_fast_approx()
                                  : sg->signal_part_derivatives();
            result.insert(result.end(), sg_derivs.begin(), sg_derivs.end());
        }
        const VALUE_TYPE theo_trash_scaled = _isolated_theo_trash_cost_scaled();
        for (LEMON_INDEX dead_end_id : dead_end_node_ids) {
            if (auto* theo = std::get_if<TheoreticalNode<intensity_type>>(&nodes[dead_end_id].get_type()))
                result.emplace_back(theo->get_spectrum_id(), theo->get_peak_index(), theo_trash_scaled);
        }
        return result;
    }

    std::vector<std::pair<size_t, double>>
    _spectrum_proportion_derivatives(bool fast) const {
        std::vector<double> accum(_no_theoretical_spectra, 0.0);
        for (const auto& sg : flow_subgraphs) {
            auto sg_derivs = fast ? sg->spectrum_proportion_derivatives_fast_approx()
                                  : sg->spectrum_proportion_derivatives();
            for (auto& [spec_id, deriv] : sg_derivs)
                accum[spec_id] += deriv;
        }
        const VALUE_TYPE theo_trash_scaled = _isolated_theo_trash_cost_scaled();
        for (size_t s = 0; s < _no_theoretical_spectra; ++s) {
            if (_isolated_theoretical_intensity[s] != 0)
                accum[s] += static_cast<double>(theo_trash_scaled) * static_cast<double>(_isolated_theoretical_intensity[s]);
        }
        std::vector<std::pair<size_t, double>> result;
        result.reserve(_no_theoretical_spectra);
        for (size_t s = 0; s < _no_theoretical_spectra; ++s)
            result.emplace_back(s, accum[s]);
        return result;
    }

public:
    // Exact per-peak / per-spectrum marginals (cheapest residual augmenting
    // route).  This is the default everything uses.
    std::vector<std::tuple<size_t, LEMON_INDEX, VALUE_TYPE>> signal_part_derivatives() const {
        return _signal_part_derivatives(/*fast=*/false);
    }
    std::vector<std::pair<size_t, double>> spectrum_proportion_derivatives() const {
        return _spectrum_proportion_derivatives(/*fast=*/false);
    }

    // Fast, APPROXIMATE variants: pure dual-potential difference, no residual
    // search.  ~O(n) vs a Dijkstra per subgraph, but the returned gradient
    // VALUES differ (lower bound on the true marginal; exact only on the
    // optimal flow support, basis-dependent at degenerate optima).  Opt-in.
    std::vector<std::tuple<size_t, LEMON_INDEX, VALUE_TYPE>>
    signal_part_derivatives_fast_approx() const {
        return _signal_part_derivatives(/*fast=*/true);
    }
    std::vector<std::pair<size_t, double>>
    spectrum_proportion_derivatives_fast_approx() const {
        return _spectrum_proportion_derivatives(/*fast=*/true);
    }

    static constexpr size_t value_type_size() {
        return sizeof(VALUE_TYPE);
    }

    static constexpr size_t index_type_size() {
        return sizeof(LEMON_INDEX);
    }

    static constexpr size_t max_value() {
        return std::numeric_limits<VALUE_TYPE>::max();
    }

    static constexpr size_t max_index() {
        return std::numeric_limits<LEMON_INDEX>::max();
    }

    // Update positions of empirical and theoretical peaks and immediately
    // re-solve each subgraph (warm-restarting NetworkSimplex if possible).
    //
    // Graph topology is fixed: only edge costs change.  Intensities are not
    // touched.  The new distributions must have the same number of peaks as
    // the originals (same peak_index range).
    //
    // For chain subgraphs (1D) the sorted position order of peaks in the
    // chain must be preserved; otherwise an exception is thrown.  If peaks
    // have genuinely crossed, rebuild the network from scratch instead.
    template<typename Distribution_t, typename DistMetric>
    void update_positions_and_solve(
        const Distribution_t* new_empirical,
        const std::vector<Distribution_t*>& new_theoretical
    ) {
        if (!built)
            throw std::runtime_error("update_positions_and_solve() must be called after build().");

        for (auto& sg_ptr : flow_subgraphs) {
            auto& sg = *sg_ptr;
            const auto& sg_nodes = sg.get_nodes();
            const auto& sg_edges = sg.get_edges();

            // Option B: reject position updates that would reorder chain nodes.
            // _build_chain_topology() walks from the lowest-ID endpoint, so the
            // chain order is deterministic.  Valid updates keep the sequence
            // monotone; a non-monotone result means peaks have genuinely crossed
            // and the topology is no longer valid.
            // nodes[i].get_id() == i by construction, so sg_nodes[nid] is a direct lookup.
            const auto& chain_order = sg.get_chain_order();
            if (chain_order.size() >= 2) {
                auto& chain_pos = sg.chain_pos_scratch();
                chain_pos.clear();
                for (LEMON_INDEX nid : chain_order) {
                    const auto& ntype = sg_nodes[nid].get_type();
                    if (const auto* emp = std::get_if<EmpiricalNode<intensity_type>>(&ntype))
                        chain_pos.push_back(new_empirical->get_point(emp->get_peak_index())[0]);
                    else if (const auto* theo = std::get_if<TheoreticalNode<intensity_type>>(&ntype))
                        chain_pos.push_back(new_theoretical[theo->get_spectrum_id()]->get_point(theo->get_peak_index())[0]);
                    // source/sink never appear in chain_order; skip anything else
                }
                bool all_nondec = true, all_noninc = true;
                for (size_t k = 1; k < chain_pos.size(); ++k) {
                    if (chain_pos[k] < chain_pos[k - 1]) all_nondec = false;
                    if (chain_pos[k] > chain_pos[k - 1]) all_noninc = false;
                }
                if (!all_nondec && !all_noninc)
                    throw std::invalid_argument(
                        "update_positions_and_solve(): new positions violate the chain's sorted "
                        "order (peaks have crossed). Rebuild the network for the new positions.");
            }

            // Compute new edge costs using the pre-allocated scratch buffer.
            auto& new_costs = sg.costs_scratch();
            std::fill(new_costs.begin(), new_costs.end(), VALUE_TYPE(0));
            for (LEMON_INDEX ii = 0; ii < static_cast<LEMON_INT>(sg_edges.size()); ++ii) {
                const auto& edge = sg_edges[ii];
                if (std::holds_alternative<MatchingEdge>(edge.get_type())) {
                    const auto& emp_t  = std::get<EmpiricalNode<intensity_type>>(edge.get_start_node().get_type());
                    const auto& theo_t = std::get<TheoreticalNode<intensity_type>>(edge.get_end_node().get_type());
                    const double d = DistMetric::dist(
                        new_empirical->get_point(emp_t.get_peak_index()),
                        new_theoretical[theo_t.get_spectrum_id()]->get_point(theo_t.get_peak_index()));
                    const double real_cost = (_p_order == 1.0) ? d : std::pow(d, _p_order);
                    // Reuse the fixed build-time scale so the warm-restarted basis
                    // stays in the same cost units.
                    new_costs[ii] = quantize_cost<VALUE_TYPE>(real_cost, _scale, _p_order == 1.0);
                } else if (std::holds_alternative<ChainEdge>(edge.get_type())) {
                    auto get_pos_1d = [&](const FlowNode<intensity_type>& n) -> double {
                        const auto& nt = n.get_type();
                        if (const auto* emp = std::get_if<EmpiricalNode<intensity_type>>(&nt))
                            return new_empirical->get_point(emp->get_peak_index())[0];
                        if (const auto* theo = std::get_if<TheoreticalNode<intensity_type>>(&nt))
                            return new_theoretical[theo->get_spectrum_id()]->get_point(theo->get_peak_index())[0];
                        throw std::runtime_error("update_positions_and_solve(): chain edge connects non-peak node.");
                    };
                    const double gap = std::abs(get_pos_1d(edge.get_start_node()) - get_pos_1d(edge.get_end_node()));
                    // Chain implies p == 1 (scale == 1): truncate, matching build.
                    new_costs[ii] = quantize_cost<VALUE_TYPE>(gap, _scale, _p_order == 1.0);
                }
                // All other edge types (SrcToEmpirical, TheoreticalToSink, trash, …)
                // have position-independent costs; apply_new_costs ignores them (new_costs[ii] = 0).
            }

            sg.apply_new_costs(new_costs);
        }
        // _last_point (spectrum proportions) is unchanged — intensities are fixed.
    }

    // Runtime-dispatch variant: selects the metric policy at run time.
    template<typename Distribution_t>
    void update_positions_and_solve(
        const Distribution_t* new_empirical,
        const std::vector<Distribution_t*>& new_theoretical,
        DistanceMetric metric
    ) {
        if (metric == DistanceMetric::L1)
            update_positions_and_solve<Distribution_t, L1Metric>(new_empirical, new_theoretical);
        else if (metric == DistanceMetric::L2)
            update_positions_and_solve<Distribution_t, L2Metric>(new_empirical, new_theoretical);
        else if (metric == DistanceMetric::LINF)
            update_positions_and_solve<Distribution_t, LinfMetric>(new_empirical, new_theoretical);
        else
            throw std::runtime_error("update_positions_and_solve(): unsupported distance metric.");
    }

    // Layer 1 (span sink): update positions, re-solve, accumulate gradients into
    // caller-owned zero-initialised spans.  emp_grad is [N_emp * DIM] row-major;
    // theo_grads[s] is [N_s * DIM] row-major.  Chain (1D) subgraphs are handled
    // via accumulate_position_gradients_chain(); dense subgraphs via accumulate_position_gradients().
    template<typename Distribution_t, typename DistMetric>
    void update_positions_and_get_gradient(
        const Distribution_t* new_empirical,
        const std::vector<Distribution_t*>& new_theoretical,
        std::span<double> emp_grad,
        std::vector<std::span<double>> theo_grads
    ) {
        static constexpr size_t DIM = std::tuple_size_v<typename Distribution_t::Point_t>;
        update_positions_and_solve<Distribution_t, DistMetric>(new_empirical, new_theoretical);
        for (auto& sg_ptr : flow_subgraphs) {
            if (sg_ptr->has_chain_edges()) {
                if constexpr (DIM == 1)
                    sg_ptr->template accumulate_position_gradients_chain<Distribution_t, DistMetric>(
                        new_empirical, new_theoretical, emp_grad, theo_grads);
                else
                    throw std::logic_error(
                        "update_positions_and_get_gradient: chain edges require DIM == 1");
            } else {
                sg_ptr->template accumulate_position_gradients<Distribution_t, DistMetric>(
                    new_empirical, new_theoretical, emp_grad, theo_grads, _p_order);
            }
        }
    }

    // Runtime-dispatch variant.
    template<typename Distribution_t>
    void update_positions_and_get_gradient(
        const Distribution_t* new_empirical,
        const std::vector<Distribution_t*>& new_theoretical,
        std::span<double> emp_grad,
        std::vector<std::span<double>> theo_grads,
        DistanceMetric metric
    ) {
        if (metric == DistanceMetric::L1)
            update_positions_and_get_gradient<Distribution_t, L1Metric>(
                new_empirical, new_theoretical, emp_grad, theo_grads);
        else if (metric == DistanceMetric::L2)
            update_positions_and_get_gradient<Distribution_t, L2Metric>(
                new_empirical, new_theoretical, emp_grad, theo_grads);
        else if (metric == DistanceMetric::LINF)
            update_positions_and_get_gradient<Distribution_t, LinfMetric>(
                new_empirical, new_theoretical, emp_grad, theo_grads);
        else
            throw std::runtime_error(
                "update_positions_and_get_gradient(): unsupported distance metric.");
    }

    // Layer 2 (vector wrapper): allocates, calls Layer 1, returns by move.
    template<typename Distribution_t, typename DistMetric>
    std::pair<std::vector<double>, std::vector<std::vector<double>>>
    update_positions_and_get_gradient(
        const Distribution_t* new_empirical,
        const std::vector<Distribution_t*>& new_theoretical
    ) {
        static constexpr size_t DIM = std::tuple_size_v<typename Distribution_t::Point_t>;
        std::vector<double> emp_grad(new_empirical->size() * DIM, 0.0);
        std::vector<std::vector<double>> theo_grads;
        theo_grads.reserve(new_theoretical.size());
        for (const auto* t : new_theoretical)
            theo_grads.emplace_back(t->size() * DIM, 0.0);
        std::vector<std::span<double>> theo_spans;
        theo_spans.reserve(new_theoretical.size());
        for (auto& v : theo_grads)
            theo_spans.emplace_back(v.data(), v.size());
        update_positions_and_get_gradient<Distribution_t, DistMetric>(
            new_empirical, new_theoretical,
            std::span<double>(emp_grad.data(), emp_grad.size()),
            theo_spans);
        return {std::move(emp_grad), std::move(theo_grads)};
    }
};



template <typename VALUE_TYPE>
class WassersteinNetworkFactory {
public:
    template<typename Distribution_t, typename DistMetric>
    static WassersteinNetwork<VALUE_TYPE, typename Distribution_t::intensity_type> create(
        const Distribution_t* empirical_spectrum,
        const std::vector<Distribution_t*>& theoretical_spectra,
        VALUE_TYPE max_dist = std::numeric_limits<VALUE_TYPE>::max(),
        double p = 1.0
    )
    {
        if (!(std::isfinite(p) && p >= 1.0))
            throw std::invalid_argument("Wasserstein order p must be a finite number >= 1.");
        using intensity_type = typename Distribution_t::intensity_type;
        // The integer-intensity backend is the bit-exact p == 1 legacy: it does
        // no cost/intensity scaling, so fractional-cost orders are unsupported.
        if constexpr (std::is_integral_v<intensity_type>)
            if (p != 1.0)
                throw std::invalid_argument(
                    "Integer intensity backend supports only p == 1; use the "
                    "double-intensity backend for p != 1.");
        double max_real_cost = 0.0;  // largest matching cost; sizes the build-time scale
        std::vector<FlowNode<intensity_type>> nodes;
        std::vector<FlowEdge<intensity_type>> edges;
        std::vector<LEMON_INDEX> dead_end_node_ids;

        static_assert(std::is_same_v<typename Distribution_t::intensity_type, intensity_type>,
                      "intensity_type does not match the intensity_type of the provided Distribution_t");
        // Empty distributions previously triggered UB inside CloserThanIter
        // (out-of-bounds access on empty sorted_indices for empty empirical,
        // and an infinite loop for empty theoretical). Reject explicitly.
        if (empirical_spectrum->size() == 0)
            throw std::invalid_argument("Empirical distribution is empty.");
        for (size_t i = 0; i < theoretical_spectra.size(); ++i)
            if (theoretical_spectra[i]->size() == 0)
                throw std::invalid_argument(
                    "Theoretical distribution at index " + std::to_string(i) + " is empty.");
        {
            size_t no_nodes = 2 + empirical_spectrum->size();
            for (auto& ts : theoretical_spectra)
                no_nodes += ts->size();
            assert_fits_lemon_index(no_nodes, "nodes");
            assert_fits_lemon_index(empirical_spectrum->size(), "empirical peaks");
            for (size_t i = 0; i < theoretical_spectra.size(); ++i)
                assert_fits_lemon_index(theoretical_spectra[i]->size(), "theoretical peaks");
            nodes.reserve(no_nodes);
        }

        // Create placeholder source and sink nodes
        nodes.emplace_back(FlowNode<intensity_type>(0, SourceNode()));
        nodes.emplace_back(FlowNode<intensity_type>(1, SinkNode()));

        for (LEMON_INDEX empirical_idx = 0; empirical_idx < static_cast<LEMON_INT>(empirical_spectrum->size()); ++empirical_idx) {
            nodes.emplace_back(FlowNode<intensity_type>(
                                    nodes.size(),
                                    EmpiricalNode(
                                        empirical_idx,
                                        empirical_spectrum->intensities()[empirical_idx])));
        }

        for (size_t theoretical_spectrum_idx = 0; theoretical_spectrum_idx < theoretical_spectra.size(); ++theoretical_spectrum_idx)
        {
            #ifdef DO_TONS_OF_PRINTS
            size_t no_processed = 0;
            size_t no_included = 0;
            std::cout << "Processing theoretical spectrum " << theoretical_spectrum_idx << " / " << theoretical_spectra.size() << std::endl;
            #endif
            const auto& theoretical_spectrum = theoretical_spectra[theoretical_spectrum_idx];

            const size_t first_theoretical_node_idx = nodes.size();
            for (LEMON_INDEX theoretical_peak_idx = 0; theoretical_peak_idx < static_cast<LEMON_INT>(theoretical_spectrum->size()); ++theoretical_peak_idx) {
                nodes.emplace_back(FlowNode<intensity_type>(
                                        nodes.size(),
                                            TheoreticalNode(
                                                theoretical_spectrum_idx,
                                                theoretical_peak_idx,
                                                theoretical_spectrum->intensities()[theoretical_peak_idx])));
            }

            // Calculate the distances between the empirical and theoretical peaks
            auto it = empirical_spectrum->template closer_than_iter<DistMetric>(*theoretical_spectrum, max_dist);
            while(it.advance())
            {
                auto [empirical_idx, theoretical_peak_idx] = it.get_indices();
                double dist = it.get_distance();
                // Order-p cost: d^p (real, unscaled).  p == 1 leaves dist unchanged
                // (pow(d,1) == d).  Quantisation to the integer solver cost happens
                // at build(), once the global scale is known.
                double real_cost = (p == 1.0) ? dist : std::pow(dist, p);
                if (real_cost > max_real_cost) max_real_cost = real_cost;
                edges.emplace_back(FlowEdge<intensity_type>(
                    edges.size(),
                    nodes[empirical_idx + 2], // +2 to skip the source and sink nodes
                    nodes[first_theoretical_node_idx + theoretical_peak_idx],
                    MatchingEdge(real_cost)
                ));
            }
        }

        std::vector<size_t> theoretical_spectra_sizes;
        theoretical_spectra_sizes.reserve(theoretical_spectra.size());
        for (const auto& theoretical_spectrum : theoretical_spectra)
            theoretical_spectra_sizes.push_back(theoretical_spectrum->size());

        assert_fits_lemon_index(edges.size(), "edges");

        return WassersteinNetwork<VALUE_TYPE, intensity_type>(
            std::move(nodes),
            std::move(edges),
            theoretical_spectra.size(),
            std::move(theoretical_spectra_sizes),
            std::move(dead_end_node_ids),
            p,
            max_real_cost
        );
    };

    template<typename Distribution_t>
    static WassersteinNetwork<VALUE_TYPE, typename Distribution_t::intensity_type> create(
        const Distribution_t* empirical_spectrum,
        const std::vector<Distribution_t*>& theoretical_spectra,
        DistanceMetric distance_metric,
        VALUE_TYPE max_dist = std::numeric_limits<VALUE_TYPE>::max(),
        double p = 1.0
    ) {
        if (distance_metric == DistanceMetric::L1) {
            return create<Distribution_t, L1Metric>(empirical_spectrum, theoretical_spectra, max_dist, p);
        } else if (distance_metric == DistanceMetric::L2) {
            return create<Distribution_t, L2Metric>(empirical_spectrum, theoretical_spectra, max_dist, p);
        } else if (distance_metric == DistanceMetric::LINF) {
            return create<Distribution_t, LinfMetric>(empirical_spectrum, theoretical_spectra, max_dist, p);
        } else {
            throw std::runtime_error("Unsupported distance metric.");
        }
    };

    // 1D chain-optimized factory. Instead of O(m·n) matching edges,
    // merges empirical and theoretical peaks into one sorted sequence and
    // emits only O(m+n) chain edges (gap-cost) between adjacent peaks.
    // In 1D, L1 = L2 = L_inf = |position difference|, so the distance
    // metric argument is accepted for API symmetry but has no effect.
    //
    // Parity with `create`: single-side fragments (runs of peaks where all
    // are empirical or all are theoretical, with no cross-side peak within
    // `max_dist`) emit no chain edges, so their nodes get zero neighbours
    // and are classified as dead-end by `split_into_subgraphs`. This
    // matches today's dense behavior of dropping unmatched mass silently.
    template<typename intensity_type_>
    static WassersteinNetwork<VALUE_TYPE, intensity_type_> create_1d(
        const VectorDistribution<1, double, intensity_type_>* empirical_spectrum,
        const std::vector<VectorDistribution<1, double, intensity_type_>*>& theoretical_spectra,
        DistanceMetric /* distance_metric */,
        VALUE_TYPE max_dist = std::numeric_limits<VALUE_TYPE>::max(),
        double p = 1.0
    ) {
        // The chain factory's gap costs are additive along the line, which only
        // equals the transport cost for p == 1 (|a-c|^p != |a-b|^p + |b-c|^p).
        if (p != 1.0)
            throw std::invalid_argument(
                "create_1d (chain factory) only supports p=1; use the dense factory for p!=1.");
        using intensity_type = intensity_type_;
        std::vector<FlowNode<intensity_type>> nodes;
        std::vector<FlowEdge<intensity_type>> edges;
        std::vector<LEMON_INDEX> dead_end_node_ids;  // recomputed in build_subgraphs()

        // Reject empty inputs for API parity with the dense `create` factory.
        if (empirical_spectrum->size() == 0)
            throw std::invalid_argument("Empirical distribution is empty.");
        for (size_t i = 0; i < theoretical_spectra.size(); ++i)
            if (theoretical_spectra[i]->size() == 0)
                throw std::invalid_argument(
                    "Theoretical distribution at index " + std::to_string(i) + " is empty.");

        // Reserve node storage (source + sink + all empirical + all theoretical).
        {
            size_t no_nodes = 2 + empirical_spectrum->size();
            for (auto& ts : theoretical_spectra)
                no_nodes += ts->size();
            assert_fits_lemon_index(no_nodes, "nodes");
            assert_fits_lemon_index(empirical_spectrum->size(), "empirical peaks");
            for (const auto& ts : theoretical_spectra)
                assert_fits_lemon_index(ts->size(), "theoretical peaks");
            nodes.reserve(no_nodes);
        }

        // Source and sink placeholders (matches dense factory's convention).
        nodes.emplace_back(FlowNode<intensity_type>(0, SourceNode()));
        nodes.emplace_back(FlowNode<intensity_type>(1, SinkNode()));

        // Position entry for the sorted merge. is_empirical tags the side.
        struct PosEntry {
            double position;
            LEMON_INDEX node_id;
            bool is_empirical;
        };
        std::vector<PosEntry> entries;
        {
            size_t no_entries = empirical_spectrum->size();
            for (const auto& ts : theoretical_spectra)
                no_entries += ts->size();
            entries.reserve(no_entries);
        }

        for (LEMON_INDEX empirical_idx = 0;
             empirical_idx < static_cast<LEMON_INT>(empirical_spectrum->size());
             ++empirical_idx) {
            nodes.emplace_back(FlowNode<intensity_type>(
                nodes.size(),
                EmpiricalNode<intensity_type>(
                    empirical_idx,
                    empirical_spectrum->intensities()[empirical_idx])));
            entries.push_back(PosEntry{
                empirical_spectrum->get_point(empirical_idx)[0],
                static_cast<LEMON_INDEX>(nodes.size() - 1),
                true});
        }

        for (size_t theoretical_spectrum_idx = 0;
             theoretical_spectrum_idx < theoretical_spectra.size();
             ++theoretical_spectrum_idx) {
            const auto& ts = theoretical_spectra[theoretical_spectrum_idx];
            for (LEMON_INDEX peak_idx = 0;
                 peak_idx < static_cast<LEMON_INT>(ts->size());
                 ++peak_idx) {
                nodes.emplace_back(FlowNode<intensity_type>(
                    nodes.size(),
                    TheoreticalNode<intensity_type>(
                        theoretical_spectrum_idx,
                        peak_idx,
                        ts->intensities()[peak_idx])));
                entries.push_back(PosEntry{
                    ts->get_point(peak_idx)[0],
                    static_cast<LEMON_INDEX>(nodes.size() - 1),
                    false});
            }
        }

        std::sort(entries.begin(), entries.end(),
                  [](const PosEntry& a, const PosEntry& b) {
                      return a.position < b.position;
                  });

        // Walk sorted entries, splitting on gaps > max_dist.
        // For each maximal run, only emit chain edges if the run contains
        // at least one empirical AND one theoretical node; otherwise, all
        // nodes in the run stay isolated and get dropped as dead-ends.
        auto flush_run = [&](size_t run_start, size_t run_end) {
            bool has_emp = false;
            bool has_theo = false;
            for (size_t i = run_start; i < run_end; ++i) {
                if (entries[i].is_empirical) has_emp = true;
                else has_theo = true;
                if (has_emp && has_theo) break;
            }
            if (!(has_emp && has_theo)) return;  // single-side run → drop

            for (size_t i = run_start + 1; i < run_end; ++i) {
                const double gap_d = entries[i].position - entries[i-1].position;
                if (gap_d > static_cast<double>(std::numeric_limits<VALUE_TYPE>::max()))
                    throw std::overflow_error(
                        "Chain gap " + std::to_string(gap_d) +
                        " overflows VALUE_TYPE (max " +
                        std::to_string(std::numeric_limits<VALUE_TYPE>::max()) + ")");
                // Store the real gap; build() quantises (p==1 => truncation, the
                // legacy behaviour).  Bidirectional: LEMON needs two arcs for flow
                // in either direction. Both carry cost = gap.
                edges.emplace_back(FlowEdge<intensity_type>(
                    edges.size(),
                    nodes[entries[i-1].node_id],
                    nodes[entries[i].node_id],
                    ChainEdge(gap_d)));
                edges.emplace_back(FlowEdge<intensity_type>(
                    edges.size(),
                    nodes[entries[i].node_id],
                    nodes[entries[i-1].node_id],
                    ChainEdge(gap_d)));
            }
        };

        if (!entries.empty()) {
            size_t run_start = 0;
            for (size_t i = 1; i < entries.size(); ++i) {
                const double gap = entries[i].position - entries[i-1].position;
                if (gap > static_cast<double>(max_dist)) {
                    flush_run(run_start, i);
                    run_start = i;
                }
            }
            flush_run(run_start, entries.size());
        }

        std::vector<size_t> theoretical_spectra_sizes;
        theoretical_spectra_sizes.reserve(theoretical_spectra.size());
        for (const auto& ts : theoretical_spectra)
            theoretical_spectra_sizes.push_back(ts->size());

        assert_fits_lemon_index(edges.size(), "edges");

        return WassersteinNetwork<VALUE_TYPE, intensity_type>(
            std::move(nodes),
            std::move(edges),
            theoretical_spectra.size(),
            std::move(theoretical_spectra_sizes),
            std::move(dead_end_node_ids)
        );
    };
};
#endif // WNET_DECOMPOSITABLE_GRAPH_HPP