#ifndef WNET_DECOMPOSITABLE_GRAPH_HPP
#define WNET_DECOMPOSITABLE_GRAPH_HPP

#include <vector>
#include <span>
#include <algorithm>
#include <unordered_map>
#include <optional>


#define LEMON_ONLY_TEMPLATES
#include <lemon/static_graph.h>
#include <lemon/network_simplex.h>

//#include "pylmcf/py_support.h"
#include "graph_elements.hpp"
#include "distribution.hpp"

#include <iostream>


template <typename VALUE_TYPE>
class WassersteinNetworkSubgraph {
    std::vector<FlowNode> nodes;
    std::vector<FlowEdge> edges;
    lemon::StaticDigraph lemon_graph;
    lemon::StaticDigraph::NodeMap<VALUE_TYPE> node_supply_map;
    lemon::StaticDigraph::ArcMap<VALUE_TYPE> capacities_map;
    lemon::StaticDigraph::ArcMap<VALUE_TYPE> costs_map;
    std::optional<lemon::NetworkSimplex<lemon::StaticDigraph, VALUE_TYPE, VALUE_TYPE>> solver;
    LEMON_INDEX simple_trash_idx;
    VALUE_TYPE empirical_intensity;
    VALUE_TYPE theoretical_intensity;
    const size_t no_target_distributions;

public:
    WassersteinNetworkSubgraph(
        const std::vector<LEMON_INDEX>& subgraph_node_ids,
        const std::vector<FlowNode>& all_nodes,
        const std::vector<FlowEdge*>& my_edges,
        size_t no_target_distributions_
    ) :
        lemon_graph(),
        node_supply_map(lemon_graph),
        capacities_map(lemon_graph),
        costs_map(lemon_graph),
        solver(),
        simple_trash_idx(std::numeric_limits<LEMON_INDEX>::max()),
        empirical_intensity(0),
        theoretical_intensity(0),
        no_target_distributions(no_target_distributions_)
    {
        nodes.reserve(subgraph_node_ids.size()+2);
        nodes.push_back(FlowNode(0, SourceNode()));
        nodes.push_back(FlowNode(1, SinkNode()));
        auto& source_node = nodes[0];
        auto& sink_node = nodes[1];

        std::unordered_map<LEMON_INDEX, LEMON_INDEX> node_id_map;

        for (const auto& node_id : subgraph_node_ids)
        {
            node_id_map[node_id] = nodes.size();
            const FlowNodeType& node_type = all_nodes[node_id].get_type();
            nodes.push_back(FlowNode(nodes.size(), node_type));
            auto& new_node = nodes.back();
            if(std::holds_alternative<EmpiricalNode>(node_type))
            {
                edges.emplace_back(
                    edges.size(),
                    source_node,
                    new_node,
                    SrcToEmpiricalEdge()
                );
            }
            else if(std::holds_alternative<TheoreticalNode>(node_type))
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

        for (const FlowEdge* edge : my_edges)
        {
            const FlowNode& start_node = edge->get_start_node();
            const auto start_node_it = node_id_map.find(start_node.get_id());
            if (start_node_it == node_id_map.end()) throw std::runtime_error("Start node of edge not found in subgraph nodes.");
            const FlowNode& end_node = edge->get_end_node();
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
        edges.emplace_back(
            edges.size(),
            nodes[0],
            nodes[1],
            SimpleTrashEdge(cost)
        );
    }

    void build() {
        edges = std::move(sorted_copy(edges, [](const FlowEdge& a, const FlowEdge& b) {
            if(a.get_start_node_id() != b.get_start_node_id())
                return a.get_start_node_id() < b.get_start_node_id();
            return a.get_end_node_id() < b.get_end_node_id();
        }));
        std::vector<std::pair<LEMON_INDEX, LEMON_INDEX>> arcs;
        arcs.reserve(edges.size());
        for (const FlowEdge& edge : edges)
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
                    else { throw std::runtime_error("Invalid FlowEdgeType"); };
                }, edges[ii].get_type());

        for (LEMON_INDEX ii = 0; ii < static_cast<LEMON_INT>(edges.size()); ++ii)
        {
            capacities_map[lemon_graph.arcFromId(ii)] = std::visit([&](const auto& arg) {
                    using T = std::decay_t<decltype(arg)>;
                    if constexpr (std::is_same_v<T, MatchingEdge>) return (VALUE_TYPE) 0;
                    else if constexpr (std::is_same_v<T, SrcToEmpiricalEdge>) {
                        VALUE_TYPE intensity = (VALUE_TYPE) std::get<EmpiricalNode>(edges[ii].get_end_node().get_type()).get_intensity();
                        empirical_intensity += intensity;
                        return intensity;
                    }
                    else if constexpr (std::is_same_v<T, TheoreticalToSinkEdge>) return (VALUE_TYPE) 0;
                    else if constexpr (std::is_same_v<T, SimpleTrashEdge>) return (VALUE_TYPE) 0;
                    else { throw std::runtime_error("Invalid FlowEdgeType"); };
                }, edges[ii].get_type());
        }
        //solver.emplace(lemon_graph);//lemon::NetworkSimplex<lemon::StaticDigraph>(lemon_graph);
        //solver->upperMap(capacities_map);
    }

    void set_point(const std::vector<double>& point) {
        if(point.size() != no_target_distributions)
            throw std::runtime_error("Point dimension: " + std::to_string(point.size()) + " does not match number of target distributions: " + std::to_string(no_target_distributions));
        theoretical_intensity = 0;
        for (LEMON_INDEX ii = 0; ii < static_cast<LEMON_INT>(edges.size()); ++ii)
        {
            const FlowEdge& edge = edges[ii];
            std::visit([&](const auto& arg) {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, MatchingEdge>) {
                    const auto& theoretical_node_type = std::get<TheoreticalNode>(edge.get_end_node().get_type());
                    capacities_map[lemon_graph.arcFromId(ii)] = (VALUE_TYPE) std::min<double>(
                        theoretical_node_type.get_intensity() * point[theoretical_node_type.get_spectrum_id()],
                        std::get<EmpiricalNode>(edge.get_start_node().get_type()).get_intensity());
                    }
                else if constexpr (std::is_same_v<T, TheoreticalToSinkEdge>) {
                    const auto& theoretical_node_type = std::get<TheoreticalNode>(edge.get_start_node().get_type());
                    VALUE_TYPE intensity = (VALUE_TYPE) (theoretical_node_type.get_intensity() * point[theoretical_node_type.get_spectrum_id()]);
                    lemon_graph.arcFromId(ii);
                    capacities_map[lemon_graph.arcFromId(ii)] = intensity;
                    theoretical_intensity += intensity;
                }
                else if constexpr (std::is_same_v<T, SrcToEmpiricalEdge>) {}
                else if constexpr (std::is_same_v<T, SimpleTrashEdge>) {}
                else { throw std::runtime_error("Invalid FlowEdgeType"); };
            }, edge.get_type());
        }
        const VALUE_TYPE total_flow = std::max<VALUE_TYPE>(empirical_intensity, theoretical_intensity);
        if(simple_trash_idx != std::numeric_limits<LEMON_INDEX>::max())
        {
            capacities_map[lemon_graph.arcFromId(simple_trash_idx)] = total_flow;
            costs_map[lemon_graph.arcFromId(simple_trash_idx)] = std::get<SimpleTrashEdge>(edges[simple_trash_idx].get_type()).get_cost();
        }
        node_supply_map[lemon_graph.nodeFromId(0)] = total_flow;
        node_supply_map[lemon_graph.nodeFromId(1)] = -total_flow;
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
                      std::to_string(solver->flow(lemon_graph.arcFromId(ii))) + "\n";
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
                      std::to_string(solver->flow(lemon_graph.arcFromId(ii))) + "\n";
        }
        return result;
    };

    size_t no_nodes() const {
        return nodes.size();
    };

    size_t no_edges() const {
        return edges.size();
    };

    const std::vector<FlowNode>& get_nodes() const {
        return nodes;
    };

    const std::vector<FlowEdge>& get_edges() const {
        return edges;
    };

    void flows_for_target(size_t spectrum_id,
                            std::vector<LEMON_INDEX>& empirical_peak_indices,
                            std::vector<LEMON_INDEX>& theoretical_peak_indices,
                            std::vector<VALUE_TYPE>& flows) const
    {
        for (LEMON_INDEX ii = 0; ii < static_cast<LEMON_INT>(edges.size()); ++ii)
        {
            const FlowEdge& edge = edges[ii];
            const VALUE_TYPE flow = solver->flow(lemon_graph.arcFromId(ii));
            if (flow == 0) continue;
            std::visit([&](const auto& arg) {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, MatchingEdge>) {
                    const auto& theoretical_node_type = std::get<TheoreticalNode>(edge.get_end_node().get_type());
                    if(theoretical_node_type.get_spectrum_id() == spectrum_id)
                    {
                        empirical_peak_indices.push_back(std::get<EmpiricalNode>(edge.get_start_node().get_type()).get_peak_index());
                        theoretical_peak_indices.push_back(theoretical_node_type.get_peak_index());
                        flows.push_back(flow);
                    }
                }
                else if constexpr (std::is_same_v<T, TheoreticalToSinkEdge>) {}
                else if constexpr (std::is_same_v<T, SrcToEmpiricalEdge>) {}
                else if constexpr (std::is_same_v<T, SimpleTrashEdge>) {}
                else { throw std::runtime_error("Invalid FlowEdgeType"); };
            }, edge.get_type());
        }
    };

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
        const double denominator = count_nodes_of_type<EmpiricalNode>() * count_nodes_of_type<TheoreticalNode>();
        return nominator / denominator;
    }

    std::vector<size_t> theoretical_spectra_involved() const {
        std::unique_ptr<bool[]> involved = std::make_unique<bool[]>(no_target_distributions);
        std::fill(involved.get(), involved.get() + no_target_distributions, false);
        for (const auto& node : nodes)
        {
            if (auto node_type = std::get_if<TheoreticalNode>(&node.get_type()))
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
};

template <typename VALUE_TYPE, typename Distribution_t>
class WassersteinNetwork {
    std::vector<FlowNode> nodes;
    std::vector<FlowEdge> edges;

    const size_t _no_theoretical_spectra;

    std::vector<LEMON_INDEX> dead_end_node_ids;
    std::vector<std::unique_ptr<WassersteinNetworkSubgraph<VALUE_TYPE>>> flow_subgraphs;

    bool built = false;

public:
    WassersteinNetwork(
    const Distribution_t* empirical_spectrum,
    const std::vector<Distribution_t*>& theoretical_spectra,
    const Distribution_t::distance_fun_t dist_fun,
    VALUE_TYPE max_dist = std::numeric_limits<VALUE_TYPE>::max()
    ) :
    _no_theoretical_spectra(theoretical_spectra.size())
    {
        {
            size_t no_nodes = 2 + empirical_spectrum->size();
            for (auto& ts : theoretical_spectra)
                no_nodes += ts->size();
            nodes.reserve(no_nodes);
        }

        // Create placeholder source and sink nodes
        nodes.emplace_back(FlowNode(0, SourceNode()));
        nodes.emplace_back(FlowNode(1, SinkNode()));

        for (LEMON_INDEX empirical_idx = 0; empirical_idx < static_cast<LEMON_INT>(empirical_spectrum->size()); ++empirical_idx) {
            nodes.emplace_back(FlowNode(
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
                nodes.emplace_back(FlowNode(
                                        nodes.size(),
                                            TheoreticalNode(
                                                theoretical_spectrum_idx,
                                                theoretical_peak_idx,
                                                theoretical_spectrum->intensities[theoretical_peak_idx])));
                const auto& theoretical_node = nodes.back();

                // Calculate the distance between the empirical and theoretical peaks
                auto [indices, distances] = empirical_spectrum->closer_than(
                    theoretical_spectrum->get_point(theoretical_peak_idx),
                    dist_fun,
                    max_dist
                );
                #ifdef DO_TONS_OF_PRINTS
                no_processed += theoretical_spectrum->size();
                no_included += indices.size();
                std::cout << no_included << " / " << no_processed << " = " << static_cast<float>(no_included) / static_cast<float>(no_processed) << std::endl;
                #endif

                for (LEMON_INDEX ii = 0; ii < static_cast<LEMON_INT>(indices.size()); ++ii)
                    edges.emplace_back(FlowEdge(
                        edges.size(),
                        nodes[indices[ii] + 2], // +2 to skip the source and sink nodes
                        theoretical_node,
                        MatchingEdge(distances[ii])
                    ));
            }
        }
        build_subgraphs();
    };

    WassersteinNetwork(const WassersteinNetwork&) = delete;
    WassersteinNetwork& operator=(const WassersteinNetwork&) = delete;
    WassersteinNetwork(WassersteinNetwork&&) = delete;
    WassersteinNetwork& operator=(WassersteinNetwork&&) = delete;

    size_t no_nodes() const {
        return nodes.size();
    };
    size_t no_edges() const {
        return edges.size();
    };
    size_t no_theoretical_spectra() const {
        return _no_theoretical_spectra;
    };

    const std::vector<FlowNode>& get_nodes() const {
        return nodes;
    };
    const std::vector<FlowEdge>& get_edges() const {
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

        std::vector<std::vector<FlowEdge*>> subgraph_edges(_subgraphs.size());
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
            flow_subgraphs.emplace_back(std::make_unique<WassersteinNetworkSubgraph<VALUE_TYPE>>(
                    _subgraphs[subgraph_idx],
                    nodes,
                    subgraph_edges[subgraph_idx],
                    _no_theoretical_spectra
            ));
        }
    }

    void add_simple_trash(VALUE_TYPE cost) {
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

    const WassersteinNetworkSubgraph<VALUE_TYPE>& get_subgraph(size_t idx) const {
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
            denominator += flow_subgraph->template count_nodes_of_type<EmpiricalNode>() * flow_subgraph->template count_nodes_of_type<TheoreticalNode>();
        return nominator / denominator;
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

#endif // WNET_DECOMPOSITABLE_GRAPH_HPP