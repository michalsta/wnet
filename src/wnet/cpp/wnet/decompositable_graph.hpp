#ifndef WNET_DECOMPOSITABLE_GRAPH_HPP
#define WNET_DECOMPOSITABLE_GRAPH_HPP

#include <vector>
#include <span>
#include <algorithm>
#include <unordered_map>
#include <optional>
#include <deque>
#include <variant>


#define LEMON_ONLY_TEMPLATES
#include <lemon/static_graph.h>
#include <lemon/network_simplex.h>
#include <lemon/cycle_canceling.h>
#include <lemon/cost_scaling.h>
#include <lemon/capacity_scaling.h>

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

struct NetworkSimplexConfig {
    NSPivotRule pivot = NSPivotRule::BLOCK_SEARCH;
    bool warm = true;
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


template <typename VALUE_TYPE, typename intensity_type>
class WassersteinNetworkSubgraph {
    std::vector<FlowNode<intensity_type>> nodes;
    std::vector<FlowEdge<intensity_type>> edges;
    lemon::StaticDigraph lemon_graph;
    lemon::StaticDigraph::NodeMap<VALUE_TYPE> node_supply_map;
    lemon::StaticDigraph::ArcMap<VALUE_TYPE> capacities_map;
    lemon::StaticDigraph::ArcMap<VALUE_TYPE> costs_map;
    std::optional<lemon::NetworkSimplex<lemon::StaticDigraph, VALUE_TYPE, VALUE_TYPE>> ns_solver;
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

    struct ChainTopology {
        std::vector<LEMON_INDEX> order;
        std::vector<LEMON_INDEX> right_arc_ids;
        std::vector<LEMON_INDEX> left_arc_ids;
        std::vector<VALUE_TYPE>  gap_cost;
        std::unordered_map<LEMON_INDEX, size_t> node_to_pos;
    };
    std::optional<ChainTopology> _chain_topo;

    bool _solver_has_value() const {
        if (std::holds_alternative<NetworkSimplexConfig>(_config)) return ns_solver.has_value();
        if (std::holds_alternative<CycleCancelingConfig>(_config)) return cc_solver.has_value();
        if (std::holds_alternative<CostScalingConfig>(_config))    return cs_solver.has_value();
        return cap_solver.has_value();
    }
    VALUE_TYPE _solver_flow(lemon::StaticDigraph::Arc arc) const {
        if (std::holds_alternative<NetworkSimplexConfig>(_config)) return ns_solver->flow(arc);
        if (std::holds_alternative<CycleCancelingConfig>(_config)) return cc_solver->flow(arc);
        if (std::holds_alternative<CostScalingConfig>(_config))    return cs_solver->flow(arc);
        return cap_solver->flow(arc);
    }
    VALUE_TYPE _solver_total_cost() const {
        if (std::holds_alternative<NetworkSimplexConfig>(_config)) return ns_solver->totalCost();
        if (std::holds_alternative<CycleCancelingConfig>(_config)) return cc_solver->totalCost();
        if (std::holds_alternative<CostScalingConfig>(_config))    return cs_solver->totalCost();
        return cap_solver->totalCost();
    }

    void _build_chain_topology() {
        std::unordered_map<LEMON_INDEX, std::vector<std::pair<LEMON_INDEX, LEMON_INDEX>>> chain_adj;
        for (LEMON_INDEX ii = 0; ii < static_cast<LEMON_INT>(edges.size()); ++ii) {
            if (std::holds_alternative<ChainEdge>(edges[ii].get_type())) {
                const LEMON_INDEX u = edges[ii].get_start_node_id();
                const LEMON_INDEX v = edges[ii].get_end_node_id();
                chain_adj[u].push_back({v, ii});
            }
        }
        if (chain_adj.empty()) return;

        LEMON_INDEX start_node = -1;
        for (const auto& [node_id, adj] : chain_adj) {
            if (adj.size() == 1) { start_node = node_id; break; }
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
                using LemonPR = lemon::NetworkSimplex<lemon::StaticDigraph, VALUE_TYPE, VALUE_TYPE>::PivotRule;
                const auto pivot = static_cast<LemonPR>(cfg.pivot);
                if (cfg.warm && ns_solver.has_value()) {
                    // Warm start: reuse the existing solver and its spanning-tree
                    // basis.  Only capacities and supplies change between calls
                    // (costs are fixed at build() time), so warmRun() can repair
                    // the previous optimal basis and reach the new optimum with
                    // far fewer pivots.  Falls back to cold start automatically
                    // if the basis becomes infeasible under the new bounds.
                    ns_solver->upperMap(capacities_map);
                    ns_solver->supplyMap(node_supply_map);
                    if (costs_changed) ns_solver->costMap(costs_map);
                    ns_solver->warmRun(pivot);
                } else {
                    ++_cold_starts_via_run;
                    ns_solver.emplace(lemon_graph);
                    ns_solver->upperMap(capacities_map);
                    ns_solver->costMap(costs_map);
                    ns_solver->supplyMap(node_supply_map);
                    ns_solver->run(pivot);
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

        std::unordered_map<LEMON_INDEX, LEMON_INDEX> node_id_map;

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
            const auto start_node_it = node_id_map.find(start_node.get_id());
            if (start_node_it == node_id_map.end()) throw std::runtime_error("Start node of edge not found in subgraph nodes.");
            const FlowNode<intensity_type>& end_node = edge->get_end_node();
            const auto end_node_it = node_id_map.find(end_node.get_id());
            if (end_node_it == node_id_map.end()) throw std::runtime_error("End node of edge not found in subgraph nodes.");
            edges.emplace_back(
                    edges.size(),
                    nodes[start_node_it->second],
                    nodes[end_node_it->second],
                    edge->get_type()
            );
        }
    }

    WassersteinNetworkSubgraph(const WassersteinNetworkSubgraph&) = delete;
    WassersteinNetworkSubgraph& operator=(const WassersteinNetworkSubgraph&) = delete;
    WassersteinNetworkSubgraph(WassersteinNetworkSubgraph&&) = delete;
    WassersteinNetworkSubgraph& operator=(WassersteinNetworkSubgraph&&) = delete;

    void add_simple_trash(VALUE_TYPE cost) {
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

    void add_experimental_trash(VALUE_TYPE cost) {
        if (simple_trash_added)
            throw std::runtime_error("add_experimental_trash() is exclusive with simple trash.");
        if (experimental_trash_added)
            throw std::runtime_error("Experimental trash already added.");
        if (built)
            throw std::runtime_error("add_experimental_trash() must be called before build().");
        // One EmpiricalTrashEdge per empirical node: EmpiricalNode -> Sink.
        // Capacity is set to max/2 in build(); see comment there.
        for (const auto& node : nodes) {
            if (!std::holds_alternative<EmpiricalNode<intensity_type>>(node.get_type())) continue;
            edges.emplace_back(edges.size(), node, nodes[1], EmpiricalTrashEdge(cost));
        }
        experimental_trash_added = true;
    }

    void add_theoretical_trash(VALUE_TYPE cost) {
        if (simple_trash_added)
            throw std::runtime_error("add_theoretical_trash() is exclusive with simple trash.");
        if (theoretical_trash_added)
            throw std::runtime_error("Theoretical trash already added.");
        if (built)
            throw std::runtime_error("add_theoretical_trash() must be called before build().");
        // One TheoreticalTrashEdge per theoretical node: Source -> TheoreticalNode.
        // Capacity is set to max/2 in build(); see comment there.
        for (const auto& node : nodes) {
            if (!std::holds_alternative<TheoreticalNode<intensity_type>>(node.get_type())) continue;
            edges.emplace_back(edges.size(), nodes[0], node, TheoreticalTrashEdge(cost));
        }
        theoretical_trash_added = true;
    }

    VALUE_TYPE simple_trash_cost() const {
        if (simple_trash_idx == std::numeric_limits<LEMON_INDEX>::max())
            throw std::runtime_error("Simple trash edge not added.");
        return std::visit([](const auto& arg) -> VALUE_TYPE {
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
            costs_map[lemon_graph.arcFromId(ii)] = std::visit([&](const auto& arg) {
                    using T = std::decay_t<decltype(arg)>;
                    if constexpr (std::is_same_v<T, MatchingEdge>) return arg.get_cost();
                    else if constexpr (std::is_same_v<T, SrcToEmpiricalEdge>) return (VALUE_TYPE) 0;
                    else if constexpr (std::is_same_v<T, TheoreticalToSinkEdge>) return (VALUE_TYPE) 0;
                    else if constexpr (std::is_same_v<T, SimpleTrashEdge>) { simple_trash_idx = ii; return arg.get_cost(); }
                    else if constexpr (std::is_same_v<T, ChainEdge>) return arg.get_cost();
                    else if constexpr (std::is_same_v<T, EmpiricalTrashEdge>) return arg.get_cost();
                    else if constexpr (std::is_same_v<T, TheoreticalTrashEdge>) return arg.get_cost();
                    else { throw std::runtime_error("Invalid FlowEdgeType"); };
                }, edges[ii].get_type());

        for (LEMON_INDEX ii = 0; ii < static_cast<LEMON_INT>(edges.size()); ++ii)
        {
            capacities_map[lemon_graph.arcFromId(ii)] = std::visit([&](const auto& arg) {
                    using T = std::decay_t<decltype(arg)>;
                    if constexpr (std::is_same_v<T, MatchingEdge>) return (VALUE_TYPE) 0;
                    else if constexpr (std::is_same_v<T, SrcToEmpiricalEdge>) {
                        VALUE_TYPE lemon_intensity = std::get<EmpiricalNode<intensity_type>>(edges[ii].get_end_node().get_type()).get_intensity();
                        lemon_empirical_intensity += lemon_intensity;
                        return lemon_intensity;
                    }
                    else if constexpr (std::is_same_v<T, TheoreticalToSinkEdge>) return (VALUE_TYPE) 0;
                    else if constexpr (std::is_same_v<T, SimpleTrashEdge>) return (VALUE_TYPE) 0;
                    // Chain edges carry unlimited flow; max/2 avoids any
                    // accidental overflow when LEMON internals sum caps.
                    else if constexpr (std::is_same_v<T, ChainEdge>) return std::numeric_limits<VALUE_TYPE>::max() / 2;
                    // Asymmetric trash edges: the adjacent anchor edge is always the
                    // binding constraint (SrcToEmpiricalEdge caps empirical inflow;
                    // TheoreticalToSinkEdge caps theoretical outflow), so a redundant
                    // tight cap here adds pivot candidates without shrinking the feasible
                    // region. Use max/2 like ChainEdge and skip set_point updates.
                    else if constexpr (std::is_same_v<T, EmpiricalTrashEdge>) return std::numeric_limits<VALUE_TYPE>::max() / 2;
                    else if constexpr (std::is_same_v<T, TheoreticalTrashEdge>) return std::numeric_limits<VALUE_TYPE>::max() / 2;
                    else { throw std::runtime_error("Invalid FlowEdgeType"); };
                }, edges[ii].get_type());
        }
        ns_solver.reset();
        cc_solver.reset();
        _build_chain_topology();
        built = true;
    }

    void build(SolverConfig config = NetworkSimplexConfig{}) {
        _config = config;
        build_impl();
    }

    void set_point(const std::vector<double>& point) {
        if(point.size() != no_target_distributions)
            throw std::runtime_error("Point dimension: " + std::to_string(point.size()) + " does not match number of target distributions: " + std::to_string(no_target_distributions));
        lemon_theoretical_intensity = 0;
        for (LEMON_INDEX ii = 0; ii < static_cast<LEMON_INT>(edges.size()); ++ii)
        {
            const FlowEdge<intensity_type>& edge = edges[ii];
            std::visit([&](const auto& arg) {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, MatchingEdge>) {
                    const auto& theoretical_node_type = std::get<TheoreticalNode<intensity_type>>(edge.get_end_node().get_type());
                    capacities_map[lemon_graph.arcFromId(ii)] = (VALUE_TYPE) std::min<double>(
                        theoretical_node_type.get_intensity() * point[theoretical_node_type.get_spectrum_id()],
                        std::get<EmpiricalNode<intensity_type>>(edge.get_start_node().get_type()).get_intensity());
                    }
                else if constexpr (std::is_same_v<T, TheoreticalToSinkEdge>) {
                    const auto& theoretical_node_type = std::get<TheoreticalNode<intensity_type>>(edge.get_start_node().get_type());
                    VALUE_TYPE lemon_intensity = (VALUE_TYPE) (theoretical_node_type.get_intensity() * point[theoretical_node_type.get_spectrum_id()]);
                    capacities_map[lemon_graph.arcFromId(ii)] = lemon_intensity;
                    lemon_theoretical_intensity += lemon_intensity;
                }
                else if constexpr (std::is_same_v<T, SrcToEmpiricalEdge>) {}
                else if constexpr (std::is_same_v<T, SimpleTrashEdge>) {}
                else if constexpr (std::is_same_v<T, ChainEdge>) {}
                // Capacity fixed at max/2 in build(); no update needed (see comment there).
                else if constexpr (std::is_same_v<T, EmpiricalTrashEdge>) {}
                else if constexpr (std::is_same_v<T, TheoreticalTrashEdge>) {}
                else { throw std::runtime_error("Invalid FlowEdgeType"); };
            }, edge.get_type());
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
            throw std::runtime_error(
                "wnet: solve() without trash edges is not supported. "
                "Without trash edges the MCF may be infeasible (sparse matching "
                "graph) or cause signed-integer overflow UB in NetworkSimplex "
                "(ART_COST = 2^62 potential accumulation). "
                "Call add_simple_trash() before build() to fix this."
            );
        }
        if(simple_trash_idx != std::numeric_limits<LEMON_INDEX>::max())
        {
            capacities_map[lemon_graph.arcFromId(simple_trash_idx)] = lemon_total_flow;
            costs_map[lemon_graph.arcFromId(simple_trash_idx)] = std::get<SimpleTrashEdge>(edges[simple_trash_idx].get_type()).get_cost();
        }
        node_supply_map[lemon_graph.nodeFromId(0)] = lemon_total_flow;
        node_supply_map[lemon_graph.nodeFromId(1)] = -lemon_total_flow;
        _run_solver();
    }

    VALUE_TYPE total_cost() const {
        if(!_solver_has_value()) throw std::runtime_error("You must call build() and set_point() before calling total_cost().");
        return _solver_total_cost();
    };

    int warm_start_count() const {
        return ns_solver.has_value() ? ns_solver->warmStartCount() : 0;
    }
    int cold_start_count() const {
        return _cold_starts_via_run +
               (ns_solver.has_value() ? ns_solver->coldStartCount() : 0);
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

    void flows_for_target(size_t spectrum_id,
                            std::vector<LEMON_INDEX>& empirical_peak_indices,
                            std::vector<LEMON_INDEX>& theoretical_peak_indices,
                            std::vector<VALUE_TYPE>& flows) const
    {
        bool has_chain = false;
        for (LEMON_INDEX ii = 0; ii < static_cast<LEMON_INT>(edges.size()); ++ii)
        {
            const FlowEdge<intensity_type>& edge = edges[ii];
            const VALUE_TYPE flow = _solver_flow(lemon_graph.arcFromId(ii));
            if (std::holds_alternative<ChainEdge>(edge.get_type())) {
                has_chain = true;
                continue;
            }
            if (flow == 0) continue;
            std::visit([&](const auto& arg) {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, MatchingEdge>) {
                    const auto& theoretical_node_type = std::get<TheoreticalNode<intensity_type>>(edge.get_end_node().get_type());
                    if(theoretical_node_type.get_spectrum_id() == spectrum_id)
                    {
                        empirical_peak_indices.push_back(std::get<EmpiricalNode<intensity_type>>(edge.get_start_node().get_type()).get_peak_index());
                        theoretical_peak_indices.push_back(theoretical_node_type.get_peak_index());
                        flows.push_back(flow);
                    }
                }
                else if constexpr (std::is_same_v<T, TheoreticalToSinkEdge>) {}
                else if constexpr (std::is_same_v<T, SrcToEmpiricalEdge>) {}
                else if constexpr (std::is_same_v<T, SimpleTrashEdge>) {}
                else if constexpr (std::is_same_v<T, ChainEdge>) {}
                else if constexpr (std::is_same_v<T, EmpiricalTrashEdge>) {}
                else if constexpr (std::is_same_v<T, TheoreticalTrashEdge>) {}
                else { throw std::runtime_error("Invalid FlowEdgeType"); };
            }, edge.get_type());
        }
        if (has_chain)
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
    std::vector<VALUE_TYPE> chain_residual_distances(LEMON_INDEX source_id) const {
        if (!_chain_topo.has_value())
            throw std::runtime_error(
                "chain_residual_distances() called on non-chain subgraph.");
        const auto& topo = *_chain_topo;
        const size_t K = topo.order.size();

        const LEMON_INDEX n = lemon_graph.nodeNum();
        const VALUE_TYPE INF = std::numeric_limits<VALUE_TYPE>::max();
        std::vector<VALUE_TYPE> dist(n, INF);
        dist[source_id] = 0;
        const LEMON_INDEX src_id = 0;
        const LEMON_INDEX sink_id = 1;

        // c_right[g]: cost to move topo.order[g] → topo.order[g+1] in residual.
        //   +gap if no leftward flow (forward of rightward arc), else -gap
        //   (reverse of leftward arc, since L[g] > 0 unlocks that residual).
        // c_left[g]: symmetric for the opposite direction.
        std::vector<VALUE_TYPE> c_right(K > 0 ? K - 1 : 0), c_left(K > 0 ? K - 1 : 0);
        for (size_t g = 0; g + 1 < K; ++g) {
            const VALUE_TYPE R = _solver_flow(lemon_graph.arcFromId(topo.right_arc_ids[g]));
            const VALUE_TYPE L = _solver_flow(lemon_graph.arcFromId(topo.left_arc_ids[g]));
            c_right[g] = (L > 0) ? -topo.gap_cost[g] : topo.gap_cost[g];
            c_left[g]  = (R > 0) ? -topo.gap_cost[g] : topo.gap_cost[g];
        }

        // src/sink connectivity per chain position.
        // Cost-0 arcs (SrcToEmpiricalEdge, TheoreticalToSinkEdge) use bool flags.
        // Non-zero trash arcs (EmpiricalTrashEdge, TheoreticalTrashEdge) store the
        // arc cost per position (INF = absent) plus forward/reverse availability.
        std::vector<bool> has_src_fwd(K, false), has_src_rev(K, false);
        std::vector<bool> has_sink_fwd(K, false), has_sink_rev(K, false);
        std::vector<VALUE_TYPE> exp_trash_cost(K, INF), theo_trash_cost(K, INF);
        std::vector<bool> exp_trash_fwd(K, false), exp_trash_rev(K, false);
        std::vector<bool> theo_trash_fwd(K, false), theo_trash_rev(K, false);
        for (LEMON_INDEX ii = 0; ii < static_cast<LEMON_INT>(edges.size()); ++ii) {
            const auto& et = edges[ii].get_type();
            if (std::holds_alternative<SrcToEmpiricalEdge>(et)) {
                auto it = topo.node_to_pos.find(edges[ii].get_end_node_id());
                if (it == topo.node_to_pos.end()) continue;
                auto arc = lemon_graph.arcFromId(ii);
                VALUE_TYPE flow = _solver_flow(arc), cap = capacities_map[arc];
                if (flow < cap) has_src_fwd[it->second] = true;
                if (flow > 0) has_src_rev[it->second] = true;
            } else if (std::holds_alternative<TheoreticalToSinkEdge>(et)) {
                auto it = topo.node_to_pos.find(edges[ii].get_start_node_id());
                if (it == topo.node_to_pos.end()) continue;
                auto arc = lemon_graph.arcFromId(ii);
                VALUE_TYPE flow = _solver_flow(arc), cap = capacities_map[arc];
                if (flow < cap) has_sink_fwd[it->second] = true;
                if (flow > 0) has_sink_rev[it->second] = true;
            } else if (const auto* e = std::get_if<EmpiricalTrashEdge>(&et)) {
                // EmpiricalNode → Sink (cost C_exp); start node is in the chain.
                auto it = topo.node_to_pos.find(edges[ii].get_start_node_id());
                if (it == topo.node_to_pos.end()) continue;
                auto arc = lemon_graph.arcFromId(ii);
                VALUE_TYPE flow = _solver_flow(arc), cap = capacities_map[arc];
                exp_trash_cost[it->second] = e->get_cost();
                if (flow < cap) exp_trash_fwd[it->second] = true;
                if (flow > 0)   exp_trash_rev[it->second] = true;
            } else if (const auto* t = std::get_if<TheoreticalTrashEdge>(&et)) {
                // Source → TheoreticalNode (cost C_theo); end node is in the chain.
                auto it = topo.node_to_pos.find(edges[ii].get_end_node_id());
                if (it == topo.node_to_pos.end()) continue;
                auto arc = lemon_graph.arcFromId(ii);
                VALUE_TYPE flow = _solver_flow(arc), cap = capacities_map[arc];
                theo_trash_cost[it->second] = t->get_cost();
                if (flow < cap) theo_trash_fwd[it->second] = true;
                if (flow > 0)   theo_trash_rev[it->second] = true;
            }
        }

        auto update_min = [&](VALUE_TYPE& a, VALUE_TYPE b) { if (b < a) a = b; };

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
            std::vector<VALUE_TYPE> prev = dist;
            relay();
            chain_sweep();
            if (dist == prev) break;
        }
        return dist;
    }

    bool has_chain_edges() const {
        for (const auto& edge : edges)
            if (std::holds_alternative<ChainEdge>(edge.get_type())) return true;
        return false;
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
        std::vector<std::span<double>>& theo_grads
    ) const {
        static constexpr size_t DIM = std::tuple_size_v<typename Distribution_t::Point_t>;
        for (LEMON_INDEX ii = 0; ii < static_cast<LEMON_INT>(edges.size()); ++ii) {
            const auto& edge = edges[ii];
            if (!std::holds_alternative<MatchingEdge>(edge.get_type())) continue;
            const VALUE_TYPE flow = _solver_flow(lemon_graph.arcFromId(ii));
            if (flow == 0) continue;
            const auto& emp_t  = std::get<EmpiricalNode<intensity_type>>(edge.get_start_node().get_type());
            const auto& theo_t = std::get<TheoreticalNode<intensity_type>>(edge.get_end_node().get_type());
            const size_t emp_idx  = emp_t.get_peak_index();
            const size_t theo_idx = theo_t.get_peak_index();
            const size_t spec_id  = theo_t.get_spectrum_id();
            const auto g = DistMetric::grad_x(
                new_empirical->get_point(emp_idx),
                new_theoretical[spec_id]->get_point(theo_idx));
            for (size_t d = 0; d < DIM; ++d) {
                emp_grad[emp_idx * DIM + d]             += static_cast<double>(flow) * g[d];
                theo_grads[spec_id][theo_idx * DIM + d] -= static_cast<double>(flow) * g[d];
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

        std::vector<VALUE_TYPE> R(K - 1), L(K - 1);
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
    std::vector<std::tuple<size_t, LEMON_INDEX, VALUE_TYPE>> signal_part_derivatives() const {
        if (!_solver_has_value())
            throw std::runtime_error("Must call solve() before signal_part_derivatives().");
        if (simple_trash_idx == std::numeric_limits<LEMON_INDEX>::max()
                && !experimental_trash_added && !theoretical_trash_added)
            throw std::runtime_error("signal_part_derivatives() requires trash edges.");

        const VALUE_TYPE INF = std::numeric_limits<VALUE_TYPE>::max();
        const LEMON_INDEX sink_id = 1;
        const bool supply_fixed = (lemon_empirical_intensity > lemon_theoretical_intensity);
        const bool use_chain = has_chain_edges();
        auto dist_src  = use_chain ? chain_residual_distances(0)       : bellman_ford_residual(0);
        auto dist_sink = use_chain ? chain_residual_distances(sink_id) : bellman_ford_residual(sink_id);

        // Pre-compute simple-trash adjustments (unused in asymmetric path).
        VALUE_TYPE trash_cost = 0, src_adjust = 0, sink_adjust = 0;
        if (!experimental_trash_added && !theoretical_trash_added) {
            trash_cost  = simple_trash_cost();
            src_adjust  = supply_fixed ? -trash_cost : 0;
            sink_adjust = supply_fixed ? 0 : trash_cost;
        }

        // Build theo->sink slack: capacity - flow.
        std::unordered_map<LEMON_INDEX, VALUE_TYPE> theo_sink_slack;
        for (LEMON_INDEX ii = 0; ii < static_cast<LEMON_INDEX>(edges.size()); ++ii) {
            if (std::holds_alternative<TheoreticalToSinkEdge>(edges[ii].get_type())) {
                auto arc = lemon_graph.arcFromId(ii);
                theo_sink_slack[edges[ii].get_start_node_id()] =
                    capacities_map[arc] - _solver_flow(arc);
            }
        }

        std::vector<std::tuple<size_t, LEMON_INDEX, VALUE_TYPE>> result;
        for (const auto& node : nodes) {
            auto* theo = std::get_if<TheoreticalNode<intensity_type>>(&node.get_type());
            if (!theo) continue;
            LEMON_INDEX node_id = node.get_id();

            // Slack > 0 with excess empirical: adding capacity does nothing.
            auto slack_it = theo_sink_slack.find(node_id);
            if (slack_it != theo_sink_slack.end() && slack_it->second > 0 && supply_fixed) {
                result.emplace_back(theo->get_spectrum_id(), theo->get_peak_index(), 0);
                continue;
            }

            VALUE_TYPE deriv;
            if (experimental_trash_added || theoretical_trash_added) {
                deriv = supply_fixed ? dist_sink[node_id] : dist_src[node_id];
                if (deriv == INF) deriv = 0;
            } else {
                deriv = trash_cost;
                if (dist_src[node_id] != INF)
                    deriv = std::min(deriv, dist_src[node_id] + src_adjust);
                if (dist_sink[node_id] != INF)
                    deriv = std::min(deriv, dist_sink[node_id] + sink_adjust);
            }

            result.emplace_back(theo->get_spectrum_id(), theo->get_peak_index(), deriv);
        }
        return result;
    }

    // Gradient of total cost w.r.t. scaling each spectrum's proportion.
    // Returns vector of (spectrum_id, derivative).
    // derivative = sum_i(peak_derivative_i * intensity_i) for each spectrum.
    std::vector<std::pair<size_t, VALUE_TYPE>> spectrum_proportion_derivatives() const {
        auto peak_derivs = signal_part_derivatives();

        // Build lookup: (spectrum_id, peak_index) -> derivative
        std::unordered_map<size_t, std::unordered_map<LEMON_INDEX, VALUE_TYPE>> deriv_map;
        for (auto& [spec_id, peak_idx, deriv] : peak_derivs)
            deriv_map[spec_id][peak_idx] = deriv;

        std::unordered_map<size_t, VALUE_TYPE> accum;
        for (const auto& node : nodes) {
            auto* theo = std::get_if<TheoreticalNode<intensity_type>>(&node.get_type());
            if (!theo) continue;
            accum[theo->get_spectrum_id()] +=
                deriv_map[theo->get_spectrum_id()][theo->get_peak_index()]
                * static_cast<VALUE_TYPE>(theo->get_intensity());
        }
        return {accum.begin(), accum.end()};
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
    VALUE_TYPE _isolated_exp_trash_cost = 0;
    VALUE_TYPE _isolated_theo_trash_cost = 0;
    std::vector<double> _last_point;

    bool built = false;

public:
    WassersteinNetwork(std::vector<FlowNode<intensity_type>>&& nodes_,
                       std::vector<FlowEdge<intensity_type>>&& edges_,
                       size_t no_theoretical_spectra_,
                       std::vector<size_t>&& theoretical_spectra_sizes_,
                       std::vector<LEMON_INDEX>&& dead_end_node_ids_
    ) :
    nodes(std::move(nodes_)),
    edges(std::move(edges_)),
    _no_theoretical_spectra(no_theoretical_spectra_),
    _theoretical_spectra_sizes(std::move(theoretical_spectra_sizes_)),
    dead_end_node_ids(std::move(dead_end_node_ids_))
    {
        build_subgraphs();
    };

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
        built(other.built)
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

    void add_simple_trash(VALUE_TYPE cost) {
        if (built)
            throw std::runtime_error("add_simple_trash() must be called before build(), not after.");
        _isolated_exp_trash_cost = cost;
        _isolated_theo_trash_cost = cost;
        for (auto& flow_subgraph : flow_subgraphs)
            flow_subgraph->add_simple_trash(cost);
    };

    void add_experimental_trash(VALUE_TYPE cost) {
        if (built)
            throw std::runtime_error("add_experimental_trash() must be called before build().");
        _isolated_exp_trash_cost = cost;
        for (auto& flow_subgraph : flow_subgraphs)
            flow_subgraph->add_experimental_trash(cost);
    };

    void add_theoretical_trash(VALUE_TYPE cost) {
        if (built)
            throw std::runtime_error("add_theoretical_trash() must be called before build().");
        _isolated_theo_trash_cost = cost;
        for (auto& flow_subgraph : flow_subgraphs)
            flow_subgraph->add_theoretical_trash(cost);
    };

    void build(SolverConfig config = NetworkSimplexConfig{}) {
        for (auto& flow_subgraph : flow_subgraphs)
            flow_subgraph->build(config);
        built = true;
    };

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

    VALUE_TYPE total_cost() const {
        VALUE_TYPE cost = 0;
        for (const auto& flow_subgraph : flow_subgraphs)
            cost += flow_subgraph->total_cost();
        cost += _isolated_exp_trash_cost * _isolated_empirical_intensity;
        for (size_t s = 0; s < _no_theoretical_spectra; ++s)
            cost += static_cast<VALUE_TYPE>(_isolated_theo_trash_cost * _isolated_theoretical_intensity[s] * _last_point[s]);
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
        return nominator / denominator;
    }

    std::vector<std::tuple<size_t, LEMON_INDEX, VALUE_TYPE>> signal_part_derivatives() const {
        std::vector<std::tuple<size_t, LEMON_INDEX, VALUE_TYPE>> result;
        for (const auto& sg : flow_subgraphs) {
            auto sg_derivs = sg->signal_part_derivatives();
            result.insert(result.end(), sg_derivs.begin(), sg_derivs.end());
        }
        for (LEMON_INDEX dead_end_id : dead_end_node_ids) {
            if (auto* theo = std::get_if<TheoreticalNode<intensity_type>>(&nodes[dead_end_id].get_type()))
                result.emplace_back(theo->get_spectrum_id(), theo->get_peak_index(), _isolated_theo_trash_cost);
        }
        return result;
    }

    std::vector<std::pair<size_t, VALUE_TYPE>> spectrum_proportion_derivatives() const {
        std::unordered_map<size_t, VALUE_TYPE> accum;
        for (const auto& sg : flow_subgraphs) {
            for (auto& [spec_id, deriv] : sg->spectrum_proportion_derivatives())
                accum[spec_id] += deriv;
        }
        for (size_t s = 0; s < _no_theoretical_spectra; ++s) {
            if (_isolated_theoretical_intensity[s] != 0)
                accum[s] += static_cast<VALUE_TYPE>(_isolated_theo_trash_cost * _isolated_theoretical_intensity[s]);
        }
        return {accum.begin(), accum.end()};
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

            // Build node_id -> node* map for chain-order validation.
            std::unordered_map<LEMON_INDEX, const FlowNode<intensity_type>*> node_map;
            node_map.reserve(sg_nodes.size());
            for (const auto& n : sg_nodes)
                node_map[n.get_id()] = &n;

            // Option B: reject position updates that would reorder chain nodes.
            // _build_chain_topology() may walk the chain in either direction
            // (ascending or descending), depending on which endpoint is found
            // first in unordered_map iteration.  Valid updates keep the sequence
            // monotone in the same direction; a non-monotone result means peaks
            // have genuinely crossed and the topology is no longer valid.
            const auto& chain_order = sg.get_chain_order();
            if (chain_order.size() >= 2) {
                std::vector<double> chain_pos;
                chain_pos.reserve(chain_order.size());
                for (LEMON_INDEX nid : chain_order) {
                    const auto& ntype = node_map.at(nid)->get_type();
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

            // Compute new edge costs.
            std::vector<VALUE_TYPE> new_costs(sg_edges.size(), 0);
            for (LEMON_INDEX ii = 0; ii < static_cast<LEMON_INT>(sg_edges.size()); ++ii) {
                const auto& edge = sg_edges[ii];
                if (std::holds_alternative<MatchingEdge>(edge.get_type())) {
                    const auto& emp_t  = std::get<EmpiricalNode<intensity_type>>(edge.get_start_node().get_type());
                    const auto& theo_t = std::get<TheoreticalNode<intensity_type>>(edge.get_end_node().get_type());
                    const double d = DistMetric::dist(
                        new_empirical->get_point(emp_t.get_peak_index()),
                        new_theoretical[theo_t.get_spectrum_id()]->get_point(theo_t.get_peak_index()));
                    if (d > static_cast<double>(std::numeric_limits<VALUE_TYPE>::max()))
                        throw std::overflow_error("update_positions_and_solve(): distance overflows VALUE_TYPE.");
                    new_costs[ii] = static_cast<VALUE_TYPE>(d);
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
                    if (gap > static_cast<double>(std::numeric_limits<VALUE_TYPE>::max()))
                        throw std::overflow_error("update_positions_and_solve(): chain gap overflows VALUE_TYPE.");
                    new_costs[ii] = static_cast<VALUE_TYPE>(gap);
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
    // theo_grads[s] is [N_s * DIM] row-major.  Chain (1D) subgraphs not supported.
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
                    new_empirical, new_theoretical, emp_grad, theo_grads);
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
        VALUE_TYPE max_dist = std::numeric_limits<VALUE_TYPE>::max()
    )
    {
        using intensity_type = typename Distribution_t::intensity_type;
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
                                        empirical_spectrum->intensities[empirical_idx])));
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
                                                theoretical_spectrum->intensities[theoretical_peak_idx])));
            }

            // Calculate the distances between the empirical and theoretical peaks
            auto it = empirical_spectrum->template closer_than_iter<DistMetric>(*theoretical_spectrum, max_dist);
            while(it.advance())
            {
                auto [empirical_idx, theoretical_peak_idx] = it.get_indices();
                double dist = it.get_distance();
                if (dist > static_cast<double>(std::numeric_limits<VALUE_TYPE>::max()))
                    throw std::overflow_error(
                        "Distance " + std::to_string(dist) +
                        " overflows VALUE_TYPE (max " +
                        std::to_string(std::numeric_limits<VALUE_TYPE>::max()) + ")");
                edges.emplace_back(FlowEdge<intensity_type>(
                    edges.size(),
                    nodes[empirical_idx + 2], // +2 to skip the source and sink nodes
                    nodes[first_theoretical_node_idx + theoretical_peak_idx],
                    MatchingEdge(static_cast<VALUE_TYPE>(dist))
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
            std::move(dead_end_node_ids)
        );
    };

    template<typename Distribution_t>
    static WassersteinNetwork<VALUE_TYPE, typename Distribution_t::intensity_type> create(
        const Distribution_t* empirical_spectrum,
        const std::vector<Distribution_t*>& theoretical_spectra,
        DistanceMetric distance_metric,
        VALUE_TYPE max_dist = std::numeric_limits<VALUE_TYPE>::max()
    ) {
        if (distance_metric == DistanceMetric::L1) {
            return create<Distribution_t, L1Metric>(empirical_spectrum, theoretical_spectra, max_dist);
        } else if (distance_metric == DistanceMetric::L2) {
            return create<Distribution_t, L2Metric>(empirical_spectrum, theoretical_spectra, max_dist);
        } else if (distance_metric == DistanceMetric::LINF) {
            return create<Distribution_t, LinfMetric>(empirical_spectrum, theoretical_spectra, max_dist);
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
        VALUE_TYPE max_dist = std::numeric_limits<VALUE_TYPE>::max()
    ) {
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
                    empirical_spectrum->intensities[empirical_idx])));
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
                        ts->intensities[peak_idx])));
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
                const VALUE_TYPE gap = static_cast<VALUE_TYPE>(gap_d);
                // Bidirectional: LEMON digraph needs two arcs for flow in
                // either direction. Both carry cost = gap.
                edges.emplace_back(FlowEdge<intensity_type>(
                    edges.size(),
                    nodes[entries[i-1].node_id],
                    nodes[entries[i].node_id],
                    ChainEdge(gap)));
                edges.emplace_back(FlowEdge<intensity_type>(
                    edges.size(),
                    nodes[entries[i].node_id],
                    nodes[entries[i-1].node_id],
                    ChainEdge(gap)));
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