// Build from the workspace:
// g++ -std=c++20 -O3 -DNDEBUG -Iwnet/src/wnet/cpp -I<installed-pylmcf>/cpp
//     wnet/prototypes/test_slope_dp_dual.cpp -o /tmp/test_slope_dp_dual
// Run: /tmp/test_slope_dp_dual --test [export.jsonl]
//      /tmp/test_slope_dp_dual --bench [chain.txt ...]
#include "slope_dp_dual.hpp"
#include <lemon/list_graph.h>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <random>
#include <string>

using namespace dp_dual_prototype;

// Independent general min-cost-flow oracle; no DP or recovered potentials.
static I network_simplex(const Input& in, I tau) {
    lemon::ListDigraph graph;
    auto src = graph.addNode(), sink = graph.addNode();
    std::vector<lemon::ListDigraph::Node> nodes;
    for (size_t i = 0; i < in.e.size(); ++i) nodes.push_back(graph.addNode());
    lemon::ListDigraph::ArcMap<I> cap(graph), cost(graph);
    auto add = [&](auto u, auto v, I c, I f) {
        auto a = graph.addArc(u, v); cap[a] = f; cost[a] = c;
    };
    I E = std::accumulate(in.e.begin(), in.e.end(), I(0));
    add(src, sink, 0, E);
    for (size_t i = 0; i < nodes.size(); ++i) {
        add(src, nodes[i], -tau, in.e[i]);
        add(nodes[i], sink, 0, in.t[i]);
        if (i) {
            add(nodes[i - 1], nodes[i], in.gap[i - 1], E);
            add(nodes[i], nodes[i - 1], in.gap[i - 1], E);
        }
    }
    lemon::NetworkSimplex<lemon::ListDigraph, I, I> ns(graph);
    if (ns.upperMap(cap).costMap(cost).stSupply(src, sink, E).run() != decltype(ns)::OPTIMAL)
        throw std::runtime_error("oracle failed");
    return ns.totalCost<I>();
}

template<class T> static void array(std::ostream& os, const T& values) {
    os << '[';
    for (size_t i = 0; i < values.size(); ++i) os << (i ? "," : "") << values[i];
    os << ']';
}
static Input random_chain(std::mt19937_64& rng, size_t n, I scale = 1) {
    Input in;
    in.e.resize(n); in.t.resize(n); in.gap.resize(n ? n - 1 : 0);
    for (size_t i = 0; i < n; ++i) {
        in.e[i] = rng() % 18; in.t[i] = rng() % 18;
        if (i) in.gap[i - 1] = I(rng() % 13) * scale;
    }
    in.prepare(); return in;
}
static void tests(const char* export_path) {
    std::mt19937_64 rng(859231);
    std::ofstream output;
    if (export_path) output.open(export_path);
    size_t checks = 0;
    for (int trial = 0; trial < 12000; ++trial) {
        I scale = trial % 7 == 0 ? I(1) << 40 : 1;
        Input in = random_chain(rng, rng() % 45, scale);
        I tau = I(rng() % 30) * scale;
        // Target all-colocated, absent-side, zero-reward, empty and saturated
        // chains in addition to random ties and large cost coefficients.
        if (trial % 13 == 0) {
            std::fill(in.gap.begin(), in.gap.end(), 0); in.prepare();
        }
        if (trial % 17 == 0) std::fill(in.e.begin(), in.e.end(), 0);
        if (trial % 19 == 0) std::fill(in.t.begin(), in.t.end(), 0);
        if (trial % 23 == 0) tau = 0;
        auto plain = solve<false>(in, tau);
        auto recorded = solve<true>(in, tau);
        std::vector<I> lazy;
        try { lazy = lazy_dual(in, plain); }
        catch (...) {
            std::cerr << "trial " << trial << " tau " << tau << "\ne "; array(std::cerr, in.e);
            std::cerr << "\nt "; array(std::cerr, in.t);
            std::cerr << "\ngap "; array(std::cerr, in.gap);
            std::cerr << "\nu "; array(std::cerr, plain.u);
            std::cerr << "\ny "; array(std::cerr, plain.y);
            std::cerr << "\ns "; array(std::cerr, plain.s);
            std::cerr << '\n'; throw;
        }
        auto clustered = cluster_dual(in, recorded);
        W value = validate(in, plain, lazy);
        if (validate(in, recorded, clustered) != value || value != W(network_simplex(in, tau)))
            throw std::runtime_error("oracle or recovery disagreement at trial " + std::to_string(trial));
        // The two constructions may choose different optimal duals at ties.
        // Both cuts must remain lower bounds after arbitrary supply changes.
        for (int probe = 0; probe < 2; ++probe) {
            Input other = in;
            for (size_t i = 0; i < in.e.size(); ++i) {
                other.e[i] = rng() % 25; other.t[i] = rng() % 25;
            }
            W minimum = network_simplex(other, tau);
            for (const auto* pi : {&lazy, &clustered}) {
                W lower = 0;
                for (size_t i = 0; i < pi->size(); ++i)
                    lower += W((*pi)[i]) * other.e[i] + W(-tau - (*pi)[i]) * other.t[i];
                if (static_cast<I>(lower) > static_cast<I>(minimum))
                    throw std::runtime_error("invalid global cut");
                ++checks;
            }
        }
        if (output && trial < 400 && scale == 1) {
            output << "{\"e\":"; array(output, in.e);
            output << ",\"t\":"; array(output, in.t);
            output << ",\"gap\":"; array(output, in.gap);
            output << ",\"pi_lazy\":"; array(output, lazy);
            output << ",\"pi_cluster\":"; array(output, clustered);
            output << ",\"tau\":" << tau << ",\"cost\":" << static_cast<I>(value) << "}\n";
        }
    }
    std::cout << "12000 primal-dual/oracle cases; " << checks << " global-cut probes passed\n";
}

using Clock = std::chrono::steady_clock;
static volatile I checksum = 0;
// Benchmarks include solve/destruction and recovery/validation respectively.
// Static input construction and DP cache preparation are excluded.
template<class F> static double measure(F&& f) {
    auto start = Clock::now(); I value = f();
    checksum = value;
    return std::chrono::duration<double, std::micro>(Clock::now() - start).count();
}
static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end()); return v[v.size() / 2];
}
static Input synthetic(size_t clusters, size_t members) {
    std::mt19937_64 rng(44);
    Input in;
    size_t n = clusters * members;
    in.e.resize(n); in.t.resize(n); in.gap.assign(n ? n - 1 : 0, 0);
    for (size_t c = 0; c < clusters; ++c) {
        for (size_t j = 0; j < members; ++j) {
            size_t i = c * members + j;
            in.e[i] = (j == 0 ? 100 + rng() % 250 : 0);
            in.t[i] = (members == 1 || j > 0 ? 30 + rng() % 120 : 0);
            if (i && j == 0) in.gap[i - 1] = 1 + rng() % 8;
        }
    }
    in.prepare(); return in;
}
static Input read_chain(const std::string& path, I& tau) {
    std::ifstream f(path); size_t n;
    if (!(f >> n >> tau)) throw std::runtime_error("invalid input " + path);
    Input in; in.e.resize(n); in.t.resize(n); in.gap.resize(n ? n - 1 : 0);
    I prev = 0;
    for (size_t i = 0; i < n; ++i) {
        I x;
        if (!(f >> x >> in.e[i] >> in.t[i])) throw std::runtime_error("truncated input");
        if (i) in.gap[i - 1] = x - prev;
        prev = x;
    }
    in.prepare(); return in;
}
static void bench(const std::string& name, const Input& in, I tau) {
    const int repeats = 41;
    std::vector<double> baseline, disabled, recording, lazy, clustered, simplex;
    auto check = solve<false>(in, tau); validate(in, check, lazy_dual(in, check));
    auto trace = solve<true>(in, tau); validate(in, trace, cluster_dual(in, trace));
    for (int trial = -3; trial < repeats; ++trial) {
        // Rotate order to reduce allocator/cache/frequency bias.
        double values[5] = {};
        for (int step = 0; step < 5; ++step) {
            int mode = ((trial + 5) % 5 + step) % 5;
            if (mode == 0) values[mode] = measure([&]() { auto r = solve<false>(in, tau); return static_cast<I>(r.profit_cost); });
            if (mode == 1) values[mode] = measure([&]() {
                // Runtime opt-in dispatch, as an eventual API would do. The
                // disabled branch calls the unchanged non-recording kernel.
                volatile bool enable_recording = false;
                if (enable_recording) { auto r = solve<true>(in, tau); return static_cast<I>(r.profit_cost); }
                auto r = solve<false>(in, tau); return static_cast<I>(r.profit_cost);
            });
            if (mode == 2) values[mode] = measure([&]() { auto r = solve<true>(in, tau); return static_cast<I>(r.profit_cost); });
            if (mode == 3) values[mode] = measure([&]() { auto pi = lazy_dual(in, check); return static_cast<I>(validate(in, check, pi)); });
            if (mode == 4) values[mode] = measure([&]() { auto pi = cluster_dual(in, trace); return static_cast<I>(validate(in, trace, pi)); });
        }
        if (trial >= 0) {
            baseline.push_back(values[0]); disabled.push_back(values[1]); recording.push_back(values[2]);
            lazy.push_back(values[3]); clustered.push_back(values[4]);
        }
    }
    // General network-simplex matching oracle on the same chain model. Keep
    // this small: construction is included, and these are cold solves.
    if (in.e.size() <= 12000) {
        for (int i = 0; i < 3; ++i)
            simplex.push_back(measure([&]() { return network_simplex(in, tau); }));
    }
    // five retained cluster vectors; no additional solve allocations/copies.
    size_t retained = (4 * trace.clusters.e.size() + trace.clusters.s.size()) * sizeof(I);
    std::cout << "{\"name\":\"" << name << "\",\"nodes\":" << in.e.size()
              << ",\"clusters\":" << in.start.size() - 1 << ",\"repeats\":" << repeats
              << ",\"solve_us\":" << median(baseline)
              << ",\"disabled_us\":" << median(disabled)
              << ",\"recording_us\":" << median(recording)
              << ",\"lazy_recovery_validation_us\":" << median(lazy)
              << ",\"cluster_recovery_validation_us\":" << median(clustered)
              << ",\"retained_bytes\":" << retained
              << ",\"cold_simplex_build_solve_us\":" << (simplex.empty() ? -1 : median(simplex))
              << ",\"samples_us\":{\"solve\":";
    array(std::cout, baseline); std::cout << ",\"disabled\":"; array(std::cout, disabled);
    std::cout << ",\"recording\":"; array(std::cout, recording);
    std::cout << ",\"lazy\":"; array(std::cout, lazy);
    std::cout << ",\"cluster\":"; array(std::cout, clustered);
    std::cout << "}}\n";
}
int main(int argc, char** argv) {
    try {
        if (argc > 1 && std::string(argv[1]) == "--test") {
            tests(argc > 2 ? argv[2] : nullptr); return 0;
        }
        if (argc > 1 && std::string(argv[1]) == "--bench") {
            std::cout << std::setprecision(9);
            for (size_t n : {size_t(2000), size_t(10000), size_t(100000)}) {
                bench("synthetic_unique_" + std::to_string(n), synthetic(n, 1), 40);
                bench("synthetic_shared_" + std::to_string(n), synthetic(n / 4, 4), 40);
            }
            for (int i = 2; i < argc; ++i) { I tau; auto in = read_chain(argv[i], tau); bench(argv[i], in, tau); }
            return 0;
        }
        std::cerr << "Usage: test_slope_dp_dual --test [export.jsonl] | --bench [chain.txt ...]\n";
        return 2;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n'; return 1;
    }
}
