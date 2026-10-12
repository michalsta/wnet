#include "sweep_options.hpp"
#include <iomanip>
using namespace sweep_options;
template <class T> T read(std::istream &s) {
    T value{};
    s.read(reinterpret_cast<char *>(&value), sizeof value);
    if (!s)
        throw std::runtime_error("truncated input");
    return value;
}
template <class T> std::vector<T> read_vector(std::istream &s, size_t n) {
    std::vector<T> v(n);
    s.read(reinterpret_cast<char *>(v.data()), n * sizeof(T));
    if (!s)
        throw std::runtime_error("truncated vector");
    return v;
}
template <class T> void write(std::ostream &s, T value) {
    s.write(reinterpret_cast<const char *>(&value), sizeof value);
}
template <class T> void write_vector(std::ostream &s, const std::vector<T> &v) {
    s.write(reinterpret_cast<const char *>(v.data()), v.size() * sizeof(T));
}
void export_components(const char *source, const char *dest, int p) {
    std::ifstream raw(source, std::ios::binary);
    size_t k = read<uint64_t>(raw);
    double radius = read<double>(raw), is = read<double>(raw);
    using D = VectorDistribution<1, double, double>;
    std::vector<std::unique_ptr<D>> ds;
    std::vector<std::vector<double>> masses;
    for (size_t s = 0; s <= k; ++s) {
        size_t n = read<uint64_t>(raw);
        auto xs = read_vector<double>(raw, n);
        auto ys = read_vector<double>(raw, n);
        masses.push_back(ys);
        std::vector<std::array<double, 1>> positions;
        for (double x : xs)
            positions.push_back({x});
        ds.push_back(std::make_unique<D>(positions, ys));
    }
    std::vector<D *> targets;
    for (size_t s = 1; s <= k; ++s)
        targets.push_back(ds[s].get());
    auto net = WassersteinNetworkFactory<I>::create_1d<double>(ds[0].get(), targets,
                                                               DistanceMetric::L1, 2 * radius, p);
    net.set_intensity_scale(is);
    net.set_flow_budget(2);
    net.set_cost_scaling(0);
    net.add_independent_asymmetric_trash(std::pow(radius, p) / 2, std::pow(radius, p) / 2);
    net.build(ConvexSweepConfig{});
    std::vector<Input> components;
    std::vector<std::vector<bool>> used;
    for (auto &m : masses)
        used.emplace_back(m.size(), false);
    I E = 0, T = 0, scale = net.scale_factor(), ce = std::llround(scale * std::pow(radius, p) / 2);
    for (size_t c = 0; c < net.no_subgraphs(); ++c) {
        const auto &sg = net.get_subgraph(c);
        auto &nodes = sg.get_nodes();
        auto &edges = sg.get_edges();
        std::vector<std::vector<std::pair<size_t, double>>> adj(nodes.size());
        for (auto &edge : edges)
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
            size_t prev = SIZE_MAX, current = start;
            double x = 0;
            for (;;) {
                order.push_back(current);
                positions.push_back(x);
                size_t next = SIZE_MAX;
                double gap = 0;
                for (auto [v, d] : adj[current])
                    if (v != prev) {
                        next = v;
                        gap = d;
                        break;
                    }
                if (next == SIZE_MAX)
                    break;
                prev = current;
                current = next;
                x += gap;
            }
        }
        Input in;
        in.pos = positions;
        in.p = p;
        in.scale = scale;
        in.tau = 2 * ce;
        for (size_t i : order) {
            I e = 0, t = 0;
            auto &type = nodes[i].get_type();
            if (auto emp = std::get_if<EmpiricalNode<double>>(&type)) {
                e = I(emp->get_intensity() * is);
                used[0][emp->get_peak_index()] = true;
            }
            if (auto theo = std::get_if<TheoreticalNode<double>>(&type)) {
                t = I(theo->get_intensity() * (1.0 / k) * is);
                used[1 + theo->get_spectrum_id()][theo->get_peak_index()] = true;
            }
            in.e.push_back(e);
            in.t.push_back(t);
            E += e;
            T += t;
        }
        components.push_back(std::move(in));
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
    I base = I(W(ce) * E + W(ce) * T);
    std::ofstream out(dest, std::ios::binary);
    write<uint64_t>(out, components.size());
    write<I>(out, scale);
    write<double>(out, is);
    write<I>(out, base);
    write<int32_t>(out, p);
    for (auto &in : components) {
        write<uint64_t>(out, in.pos.size());
        write<I>(out, in.tau);
        write_vector(out, in.pos);
        write_vector(out, in.e);
        write_vector(out, in.t);
    }
    std::cout << "{\"event\":\"export\",\"components\":" << components.size()
              << ",\"scale\":" << scale << ",\"empirical_mass\":" << E
              << ",\"theoretical_mass\":" << T << ",\"baseline\":" << base << "}" << std::endl;
}
int main(int argc, char **argv) {
    try {
        if (argc > 1 && std::string(argv[1]) == "export") {
            export_components(argv[2], argv[3], std::stoi(argv[4]));
            return 0;
        }
        if (argc < 4)
            throw std::invalid_argument("components mode repetitions [grid step]");
        std::ifstream input(argv[1], std::ios::binary);
        size_t count = read<uint64_t>(input);
        I scale = read<I>(input);
        double is = read<double>(input);
        I base = read<I>(input);
        int p = read<int32_t>(input);
        std::vector<Input> inputs;
        size_t positions = 0;
        for (size_t c = 0; c < count; ++c) {
            Input in;
            size_t n = read<uint64_t>(input);
            in.scale = scale;
            in.p = p;
            in.tau = read<I>(input);
            in.pos = read_vector<double>(input, n);
            in.e = read_vector<I>(input, n);
            in.t = read_vector<I>(input, n);
            positions += n;
            inputs.push_back(std::move(in));
        }
        std::string mode = argv[2];
        size_t repeats = std::stoul(argv[3]);
        double requested = argc > 4 ? std::stod(argv[4]) : 1e-6;
        std::cout << std::setprecision(17) << "{\"event\":\"start\",\"positions\":" << positions
                  << ",\"components\":" << count << ",\"scale\":" << scale << ",\"mode\":\"" << mode
                  << "\"}" << std::endl;
        std::vector<double> samples;
        W expected = 0, expected_dual = 0;
        size_t pairs = 0, matches = 0;
        I epsilon = 0;
        double hmax = 0;
        bool model_only =
            mode == "grid-model" || mode == "grid-sweep-model" || mode == "slope-native";
        long double isolated_cost = (long double)base / (scale * is), model_expected = 0;
        for (auto &in : inputs)
            isolated_cost -= (long double)in.tau *
                             (std::accumulate(in.e.begin(), in.e.end(), I(0)) +
                              std::accumulate(in.t.begin(), in.t.end(), I(0))) /
                             (2 * scale * is);
        auto bounds = [&](W profit, W dual, long double model_cost) {
            std::cout << ",\"upper\":" << double((long double)(base - profit) / (scale * is));
            if (model_only)
                std::cout << ",\"lower\":null,\"gap\":null,\"model_cost\":" << double(model_cost)
                          << ",\"model_gap\":0";
            else
                std::cout << ",\"lower\":" << double((long double)(base - dual) / (scale * is))
                          << ",\"gap\":" << double((long double)(dual - profit) / (scale * is));
        };
        for (size_t sample = 0; sample <= repeats; ++sample) {
            W profit = 0, dual = 0;
            long double model_cost = isolated_cost;
            pairs = matches = 0;
            epsilon = 0;
            hmax = 0;
            auto begin = std::chrono::steady_clock::now();
            for (auto &in : inputs) {
                Result r;
                if (mode == "baseline")
                    r = baseline(in);
                else if (mode == "cached")
                    r = cached(in);
                else if (mode == "cost-scaling")
                    r = cost_scaling(in);
                else if (mode == "active")
                    r = cost_scaling(in, 8000000, true);
                else if (mode == "approx")
                    r = approximate(in);
                else if (mode == "grid" || mode == "grid-model" || mode == "grid-sweep-model" ||
                         mode == "grid-fast") {
                    double h = requested;
                    if (p == 2) {
                        I E = std::accumulate(in.e.begin(), in.e.end(), I(0));
                        h = std::max(h, double(std::sqrt(16.L * (long double)in.tau / in.scale * E /
                                                         std::numeric_limits<I>::max())));
                    }
                    hmax = std::max(hmax, h);
                    r = monge(in, h, !model_only, mode == "grid-sweep-model", mode == "grid-fast");
                } else if (mode == "slope" || mode == "slope-native" || mode == "slope-fast")
                    r = slope_chain(in, !model_only, mode == "slope-fast");
                else
                    throw std::invalid_argument("bad mode");
                if (model_only) {
                    if (mode == "grid-model" || mode == "grid-sweep-model")
                        model_cost += r.model_cost / is;
                    else
                        model_cost += ((long double)in.tau *
                                           (std::accumulate(in.e.begin(), in.e.end(), I(0)) +
                                            std::accumulate(in.t.begin(), in.t.end(), I(0))) /
                                           2 -
                                       (long double)r.profit) /
                                      (scale * is);
                    profit += r.original_profit;
                    dual += r.original_profit;
                } else {
                    profit += r.profit;
                    dual += r.dual_profit;
                }
                pairs += r.pairs;
                matches += r.matches.size();
                epsilon = std::max(epsilon, r.epsilon);
            }
            double ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin)
                    .count();
            if (sample == 0) {
                expected = profit;
                expected_dual = dual;
                model_expected = model_cost;
            } else {
                if (profit != expected || dual != expected_dual || model_expected != model_cost)
                    throw std::runtime_error("unstable result");
                samples.push_back(ms);
            }
            std::cout << "{\"event\":\"sample\",\"sample\":" << sample << ",\"ms\":" << ms;
            bounds(profit, dual, model_cost);
            std::cout << "}" << std::endl;
        }
        std::sort(samples.begin(), samples.end());
        std::cout << "{\"event\":\"result\",\"median_ms\":" << samples[samples.size() / 2]
                  << ",\"min_ms\":" << samples.front() << ",\"max_ms\":" << samples.back();
        bounds(expected, expected_dual, model_expected);
        std::cout << ",\"pairs\":" << pairs << ",\"matches\":" << matches
                  << ",\"epsilon\":" << epsilon << ",\"grid_step\":" << hmax << "}" << std::endl;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
