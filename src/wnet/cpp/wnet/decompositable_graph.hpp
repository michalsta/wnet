#ifndef WNET_DECOMPOSITABLE_GRAPH_HPP
#define WNET_DECOMPOSITABLE_GRAPH_HPP

#include <vector>
#include <span>
#include <algorithm>
#include <unordered_map>
#include <optional>
#include <deque>


#define LEMON_ONLY_TEMPLATES
#include <lemon/static_graph.h>
#include <lemon/network_simplex.h>

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
    std::optional<lemon::NetworkSimplex<lemon::StaticDigraph, VALUE_TYPE, VALUE_TYPE>> solver;
    LEMON_INDEX simple_trash_idx;
    VALUE_TYPE lemon_empirical_intensity;
    VALUE_TYPE lemon_theoretical_intensity;
    const size_t no_target_distributions;
    bool built = false;

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
        solver(),
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
        if (simple_trash_idx != std::numeric_limits<LEMON_INDEX>::max())
            throw std::runtime_error("Simple trash edge already added.");
        if (built)
            throw std::runtime_error("add_simple_trash() must be called before build(), not after.");
        edges.emplace_back(
            edges.size(),
            nodes[0],
            nodes[1],
            SimpleTrashEdge(cost)
        );
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

    void build() {
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
                    else { throw std::runtime_error("Invalid FlowEdgeType"); };
                }, edges[ii].get_type());
        }
        //solver.emplace(lemon_graph);//lemon::NetworkSimplex<lemon::StaticDigraph>(lemon_graph);
        //solver->upperMap(capacities_map);
        solver.reset();
        built = true;
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
                    lemon_graph.arcFromId(ii);
                    capacities_map[lemon_graph.arcFromId(ii)] = lemon_intensity;
                    lemon_theoretical_intensity += lemon_intensity;
                }
                else if constexpr (std::is_same_v<T, SrcToEmpiricalEdge>) {}
                else if constexpr (std::is_same_v<T, SimpleTrashEdge>) {}
                else if constexpr (std::is_same_v<T, ChainEdge>) {}
                else { throw std::runtime_error("Invalid FlowEdgeType"); };
            }, edge.get_type());
        }
        const VALUE_TYPE lemon_total_flow = std::max<VALUE_TYPE>(lemon_empirical_intensity, lemon_theoretical_intensity);
        if(simple_trash_idx != std::numeric_limits<LEMON_INDEX>::max())
        {
            capacities_map[lemon_graph.arcFromId(simple_trash_idx)] = lemon_total_flow;
            costs_map[lemon_graph.arcFromId(simple_trash_idx)] = std::get<SimpleTrashEdge>(edges[simple_trash_idx].get_type()).get_cost();
        }
        node_supply_map[lemon_graph.nodeFromId(0)] = lemon_total_flow;
        node_supply_map[lemon_graph.nodeFromId(1)] = -lemon_total_flow;
        solver.emplace(lemon_graph);
        solver->upperMap(capacities_map);
        solver->costMap(costs_map);
        solver->supplyMap(node_supply_map);
        solver->run();
    }

    VALUE_TYPE total_cost() const {
        if(!solver) throw std::runtime_error("You must call build() and set_point() before calling total_cost().");
        return solver->totalCost();
    };

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
                      (solver.has_value() ?
                      std::to_string(solver->flow(lemon_graph.arcFromId(ii))) + "\n" :  "not yet computed\n");
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
                      (solver.has_value() ?
                      std::to_string(solver->flow(lemon_graph.arcFromId(ii))) + "\n" :  "not yet computed\n");
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
            const VALUE_TYPE flow = solver->flow(lemon_graph.arcFromId(ii));
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
        // Per chain node: outgoing chain arcs as (neighbor_node_id, arc_id).
        std::unordered_map<LEMON_INDEX, std::vector<std::pair<LEMON_INDEX, LEMON_INDEX>>> chain_adj;
        for (LEMON_INDEX ii = 0; ii < static_cast<LEMON_INT>(edges.size()); ++ii) {
            if (std::holds_alternative<ChainEdge>(edges[ii].get_type())) {
                const LEMON_INDEX u = edges[ii].get_start_node_id();
                const LEMON_INDEX v = edges[ii].get_end_node_id();
                chain_adj[u].push_back({v, ii});
            }
        }
        if (chain_adj.empty()) return;

        // Endpoints have exactly one outgoing chain arc (one neighbor).
        // Internal nodes have two. Pick any endpoint as the walk's start.
        LEMON_INDEX start_node = -1;
        for (const auto& [node_id, adj] : chain_adj) {
            if (adj.size() == 1) {
                start_node = node_id;
                break;
            }
        }
        if (start_node == -1)
            throw std::runtime_error(
                "Chain subgraph has no endpoint — malformed chain.");

        // Walk the chain. right_arc_ids[g] is the arc from chain_order[g]
        // toward chain_order[g+1] (the "walk direction"); left_arc_ids[g]
        // is its reverse arc.
        std::vector<LEMON_INDEX> chain_order;
        std::vector<LEMON_INDEX> right_arc_ids;
        std::vector<LEMON_INDEX> left_arc_ids;
        chain_order.push_back(start_node);
        LEMON_INDEX prev = -1;
        LEMON_INDEX curr = start_node;
        while (true) {
            LEMON_INDEX next = -1;
            LEMON_INDEX out_arc = -1;
            for (const auto& [neighbor, arc_id] : chain_adj[curr]) {
                if (neighbor != prev) {
                    next = neighbor;
                    out_arc = arc_id;
                    break;
                }
            }
            if (next == -1) break;
            LEMON_INDEX in_arc = -1;
            for (const auto& [neighbor, arc_id] : chain_adj[next]) {
                if (neighbor == curr) {
                    in_arc = arc_id;
                    break;
                }
            }
            if (in_arc == -1)
                throw std::runtime_error(
                    "Chain arc is not bidirectional — malformed chain.");
            chain_order.push_back(next);
            right_arc_ids.push_back(out_arc);
            left_arc_ids.push_back(in_arc);
            prev = curr;
            curr = next;
        }

        const size_t K = chain_order.size();
        if (K < 2) return;  // Isolated node — no gap flow to decompose.

        // Read per-gap forward/reverse flows from the solver.
        std::vector<VALUE_TYPE> R(K - 1), L(K - 1);
        for (size_t g = 0; g < K - 1; ++g) {
            R[g] = solver->flow(lemon_graph.arcFromId(right_arc_ids[g]));
            L[g] = solver->flow(lemon_graph.arcFromId(left_arc_ids[g]));
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
                const auto& node_type = nodes[chain_order[i]].get_type();
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
                const auto& node_type = nodes[chain_order[i]].get_type();
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
            const VALUE_TYPE flow = solver->flow(lemon_graph.arcFromId(ii));
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
        const double nominator = count_edges_of_type<MatchingEdge>();
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
        return solver.has_value();
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
                VALUE_TYPE flow = solver->flow(arc);
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
    std::vector<std::tuple<size_t, LEMON_INDEX, VALUE_TYPE>> signal_part_derivatives() const {
        if (!solver)
            throw std::runtime_error("Must call solve() before signal_part_derivatives().");
        if (simple_trash_idx == std::numeric_limits<LEMON_INDEX>::max())
            throw std::runtime_error("signal_part_derivatives() requires simple trash.");

        const VALUE_TYPE trash_cost = simple_trash_cost();
        const VALUE_TYPE INF = std::numeric_limits<VALUE_TYPE>::max();
        const LEMON_INDEX sink_id = 1;

        auto dist_src = bellman_ford_residual(0);
        auto dist_sink = bellman_ford_residual(sink_id);

        const VALUE_TYPE src_adjust = (lemon_empirical_intensity > lemon_theoretical_intensity)
            ? -trash_cost : 0;
        const VALUE_TYPE sink_adjust = (lemon_empirical_intensity > lemon_theoretical_intensity)
            ? 0 : trash_cost;

        // Build theo->sink slack: capacity - flow
        std::unordered_map<LEMON_INDEX, VALUE_TYPE> theo_sink_slack;
        for (LEMON_INDEX ii = 0; ii < static_cast<LEMON_INDEX>(edges.size()); ++ii) {
            if (std::holds_alternative<TheoreticalToSinkEdge>(edges[ii].get_type())) {
                auto arc = lemon_graph.arcFromId(ii);
                theo_sink_slack[edges[ii].get_start_node_id()] =
                    capacities_map[arc] - solver->flow(arc);
            }
        }

        std::vector<std::tuple<size_t, LEMON_INDEX, VALUE_TYPE>> result;
        for (const auto& node : nodes) {
            auto* theo = std::get_if<TheoreticalNode<intensity_type>>(&node.get_type());
            if (!theo) continue;
            LEMON_INDEX node_id = node.get_id();

            // If slack > 0 and excess empirical supply, the solver already
            // chose not to use this node -- adding capacity changes nothing.
            auto slack_it = theo_sink_slack.find(node_id);
            if (slack_it != theo_sink_slack.end() && slack_it->second > 0
                && lemon_empirical_intensity > lemon_theoretical_intensity) {
                result.emplace_back(theo->get_spectrum_id(), theo->get_peak_index(), 0);
                continue;
            }

            VALUE_TYPE deriv = trash_cost;
            if (dist_src[node_id] != INF)
                deriv = std::min(deriv, dist_src[node_id] + src_adjust);
            if (dist_sink[node_id] != INF)
                deriv = std::min(deriv, dist_sink[node_id] + sink_adjust);

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

    /*
    template<typename Distribution_t, DistanceMetric dist_fun>
    WassersteinNetwork(
    const Distribution_t* empirical_spectrum,
    const std::vector<Distribution_t*>& theoretical_spectra,
    VALUE_TYPE max_dist = std::numeric_limits<VALUE_TYPE>::max()
    ) :
    _no_theoretical_spectra(theoretical_spectra.size())
    {
        static_assert(std::is_same_v<typename Distribution_t::intensity_type, intensity_type>,
                      "intensity_type does not match the intensity_type of the provided Distribution_t");
        {
            size_t no_nodes = 2 + empirical_spectrum->size();
            for (auto& ts : theoretical_spectra)
                no_nodes += ts->size();
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

            for (LEMON_INDEX theoretical_peak_idx = 0; theoretical_peak_idx < static_cast<LEMON_INT>(theoretical_spectrum->size()); ++theoretical_peak_idx) {
                nodes.emplace_back(FlowNode<intensity_type>(
                                        nodes.size(),
                                            TheoreticalNode(
                                                theoretical_spectrum_idx,
                                                theoretical_peak_idx,
                                                theoretical_spectrum->intensities[theoretical_peak_idx])));
                const auto& theoretical_node = nodes.back();

                // Calculate the distance between the empirical and theoretical peaks
                auto it = empirical_spectrum->closer_than_iter_point(
                    theoretical_spectrum->get_point(theoretical_peak_idx),
                    dist_fun,
                    max_dist
                );
                while(it.advance())
                {
                    edges.emplace_back(FlowEdge<intensity_type>(
                        edges.size(),
                        nodes[it.get_index() + 2], // +2 to skip the source and sink nodes
                        theoretical_node,
                        MatchingEdge(static_cast<VALUE_TYPE>(it.get_distance()))
                    ));
                }
            }
        }
        build_subgraphs();
    };
*/
    // template<size_t DIM>
    // WassersteinNetwork(
    //     const VectorDistribution<DIM, double, intensity_type>* empirical_spectrum,
    //     const std::vector<VectorDistribution<DIM, double, intensity_type>*>& theoretical_spectra,
    //     DistanceMetric distance_metric,
    //     VALUE_TYPE max_dist = std::numeric_limits<VALUE_TYPE>::max()
    // ) :
    // WassersteinNetwork(from_vector_distribution<DIM>(
    //     empirical_spectrum,
    //     theoretical_spectra,
    //     distance_metric,
    //     max_dist
    // ))
    // {}
/*
    template<size_t DIM>
    static WassersteinNetwork<VALUE_TYPE, intensity_type> from_vector_distribution(
        const VectorDistribution<DIM, double, intensity_type>* empirical_spectrum,
        const std::vector<VectorDistribution<DIM, double, intensity_type>*>& theoretical_spectra,
        DistanceMetric distance_metric,
        VALUE_TYPE max_dist = std::numeric_limits<VALUE_TYPE>::max()
    ) {
        if (distance_metric == DistanceMetric::L1) {
            return WassersteinNetwork<VALUE_TYPE, intensity_type>::template WassersteinNetwork<
                VectorDistribution<DIM, double, intensity_type>,
                DistanceMetric::L1
            >(empirical_spectrum, theoretical_spectra, max_dist);
        } else if (distance_metric == DistanceMetric::L2) {
            return WassersteinNetwork<VALUE_TYPE, intensity_type>::template WassersteinNetwork<
                VectorDistribution<DIM, double, intensity_type>,
                DistanceMetric::L2
            >(empirical_spectrum, theoretical_spectra, max_dist);
        } else if (distance_metric == DistanceMetric::LINF) {
            return WassersteinNetwork<VALUE_TYPE, intensity_type>::template WassersteinNetwork<
                VectorDistribution<DIM, double, intensity_type>,
                DistanceMetric::LINF
            >(empirical_spectrum, theoretical_spectra, max_dist);
        } else {
            throw std::runtime_error("Unsupported distance metric.");
        }
    }*/
        // switch (distance_metric) {
        //     case DistanceMetric::L1:
        //         return WassersteinNetwork<l1_distance<DIM, double>>(empirical_spectrum, theoretical_spectra, max_dist);
        //     case DistanceMetric::L2:
        //         return WassersteinNetwork<l2_distance<DIM, double>>(empirical_spectrum, theoretical_spectra, max_dist);
        //     case DistanceMetric::LINF:
        //         return WassersteinNetwork<linf_distance<DIM, double>>(empirical_spectrum, theoretical_spectra, max_dist);
        //     default:
        //         throw std::runtime_error("Unsupported distance metric.");
        // }


    WassersteinNetwork(const WassersteinNetwork&) = delete;
    WassersteinNetwork& operator=(const WassersteinNetwork&) = delete;
    WassersteinNetwork(WassersteinNetwork&& other) :
        nodes(std::move(other.nodes)),
        edges(std::move(other.edges)),
        _no_theoretical_spectra(other._no_theoretical_spectra),
        _theoretical_spectra_sizes(std::move(other._theoretical_spectra_sizes)),
        dead_end_node_ids(std::move(other.dead_end_node_ids)),
        flow_subgraphs(std::move(other.flow_subgraphs)),
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
    }

    void add_simple_trash(VALUE_TYPE cost) {
        if (built)
            throw std::runtime_error("add_simple_trash() must be called before build(), not after.");
        for (auto& flow_subgraph : flow_subgraphs)
            flow_subgraph->add_simple_trash(cost);
    };

    void build() {
        for (auto& flow_subgraph : flow_subgraphs)
            flow_subgraph->build();
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

        for (auto& flow_subgraph : flow_subgraphs)
            flow_subgraph->set_point(point);
    };

    VALUE_TYPE total_cost() const {
        VALUE_TYPE total_cost = 0;
        for (const auto& flow_subgraph : flow_subgraphs)
            total_cost += flow_subgraph->total_cost();
        return total_cost;
    };

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
        const double nominator = count_edges_of_type<MatchingEdge>();
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
        return result;
    }

    std::vector<std::pair<size_t, VALUE_TYPE>> spectrum_proportion_derivatives() const {
        std::unordered_map<size_t, VALUE_TYPE> accum;
        for (const auto& sg : flow_subgraphs) {
            for (auto& [spec_id, deriv] : sg->spectrum_proportion_derivatives())
                accum[spec_id] += deriv;
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
};



template <typename VALUE_TYPE>
class WassersteinNetworkFactory {
public:
    template<typename Distribution_t, DistanceMetric dist_fun>
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
            auto it = empirical_spectrum->template closer_than_iter<dist_fun>(*theoretical_spectrum, max_dist);
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
            return create<Distribution_t, DistanceMetric::L1>(empirical_spectrum, theoretical_spectra, max_dist);
        } else if (distance_metric == DistanceMetric::L2) {
            return create<Distribution_t, DistanceMetric::L2>(empirical_spectrum, theoretical_spectra, max_dist);
        } else if (distance_metric == DistanceMetric::LINF) {
            return create<Distribution_t, DistanceMetric::LINF>(empirical_spectrum, theoretical_spectra, max_dist);
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