// Export full production factory components without solving or changing caps.
#include "fixture.hpp"
#include "wnet/decompositable_graph.hpp"
#include <cmath>
#include <iostream>
using namespace sweep_trials;
int main(int argc, char **argv) {
  if (argc != 5)
    throw std::invalid_argument(
        "export raw_input destination p explicit_scale_or_0");
  std::ifstream raw(argv[1], std::ios::binary);
  const size_t k = read<uint64_t>(raw);
  const double radius = read<double>(raw), is = read<double>(raw),
               p = std::stod(argv[3]);
  using D = VectorDistribution<1, double, double>;
  std::vector<std::unique_ptr<D>> ds;
  std::vector<std::vector<double>> masses;
  for (size_t s = 0; s <= k; ++s) {
    size_t n = read<uint64_t>(raw);
    auto xs = readv<double>(raw, n), ys = readv<double>(raw, n);
    masses.push_back(ys);
    std::vector<std::array<double, 1>> positions;
    for (double x : xs)
      positions.push_back({x});
    ds.push_back(std::make_unique<D>(positions, ys));
  }
  std::vector<D *> targets;
  for (size_t s = 1; s <= k; ++s)
    targets.push_back(ds[s].get());
  auto net = WassersteinNetworkFactory<I>::create_1d<double>(
      ds[0].get(), targets, DistanceMetric::L1, 2 * radius, p);
  net.set_intensity_scale(is);
  net.set_flow_budget(2);
  net.set_cost_scaling(std::stoll(argv[4]));
  net.add_independent_asymmetric_trash(std::pow(radius, p) / 2,
                                       std::pow(radius, p) / 2);
  net.build(ConvexSweepConfig{});
  std::vector<Input> inputs;
  std::vector<std::vector<bool>> used;
  for (const auto &m : masses)
    used.emplace_back(m.size(), false);
  I E = 0, T = 0, scale = net.scale_factor(),
    ce = std::llround(scale * std::pow(radius, p) / 2);
  for (size_t c = 0; c < net.no_subgraphs(); ++c) {
    const auto &sg = net.get_subgraph(c);
    const auto &nodes = sg.get_nodes();
    const auto &edges = sg.get_edges();
    std::vector<std::vector<std::pair<size_t, double>>> adj(nodes.size());
    for (const auto &edge : edges)
      if (std::holds_alternative<ChainEdge>(edge.get_type()))
        adj[edge.get_start_node_id()].push_back(
            {size_t(edge.get_end_node_id()), edge.get_cost()});
    size_t start = SIZE_MAX;
    for (size_t i = 0; i < adj.size(); ++i)
      if (adj[i].size() == 1) {
        start = i;
        break;
      }
    std::vector<size_t> order;
    std::vector<double> positions;
    if (start == SIZE_MAX) {
      for (size_t i = 2; i < nodes.size(); ++i) {
        order.push_back(i);
        positions.push_back(0);
      }
    } else {
      size_t prev = SIZE_MAX, cur = start;
      double x = 0;
      for (;;) {
        order.push_back(cur);
        positions.push_back(x);
        size_t next = SIZE_MAX;
        double gap = 0;
        for (auto [v, d] : adj[cur])
          if (v != prev) {
            next = v;
            gap = d;
            break;
          }
        if (next == SIZE_MAX)
          break;
        prev = cur;
        cur = next;
        x += gap;
      }
    }
    Input in;
    in.pos = positions;
    for (size_t i : order) {
      I e = 0, t = 0;
      uint8_t theo = 0;
      const auto &type = nodes[i].get_type();
      if (auto emp = std::get_if<EmpiricalNode<double>>(&type)) {
        e = I(emp->get_intensity() * is);
        used[0][emp->get_peak_index()] = true;
      }
      if (auto th = std::get_if<TheoreticalNode<double>>(&type)) {
        t = I(th->get_intensity() * (1.0 / k) * is);
        theo = 1;
        used[1 + th->get_spectrum_id()][th->get_peak_index()] = true;
      }
      in.e.push_back(e);
      in.t.push_back(t);
      in.theo.push_back(theo);
      E += e;
      T += t;
    }
    inputs.push_back(std::move(in));
  }
  for (size_t s = 0; s <= k; ++s) {
    double isolated = 0;
    for (size_t i = 0; i < masses[s].size(); ++i)
      if (!used[s][i])
        isolated += masses[s][i];
    if (s == 0)
      E += I(isolated * is);
    else
      T += I(isolated * (1.0 / k) * is);
  }
  const I base = static_cast<I>(W(ce) * E + W(ce) * T);
  std::ofstream out(argv[2], std::ios::binary);
  write<uint64_t>(out, 0x31545343);
  write<uint64_t>(out, inputs.size());
  write<I>(out, scale);
  write<double>(out, is);
  write<I>(out, ce);
  write<double>(out, p);
  write<I>(out, base);
  for (const auto &in : inputs) {
    write<uint64_t>(out, in.pos.size());
    writev(out, in.pos);
    writev(out, in.e);
    writev(out, in.t);
    writev(out, in.theo);
  }
  if (!out)
    throw std::runtime_error("fixture write failed");
  std::cout << "{\"components\":" << inputs.size() << ",\"scale\":" << scale
            << ",\"base\":" << base << ",\"p\":" << p << "}\n";
}
