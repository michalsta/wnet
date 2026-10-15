#ifndef SWEEP_TRIAL_FIXTURE_HPP
#define SWEEP_TRIAL_FIXTURE_HPP
#include "trials.hpp"
#include <fstream>
namespace sweep_trials {
template <class T> T read(std::istream &f) {
  T v{};
  f.read(reinterpret_cast<char *>(&v), sizeof v);
  if (!f)
    throw std::runtime_error("truncated fixture");
  return v;
}
template <class T> std::vector<T> readv(std::istream &f, size_t n) {
  std::vector<T> v(n);
  f.read(reinterpret_cast<char *>(v.data()), sizeof(T) * n);
  if (!f)
    throw std::runtime_error("truncated fixture vector");
  return v;
}
template <class T> void write(std::ostream &f, T v) {
  f.write(reinterpret_cast<const char *>(&v), sizeof v);
}
template <class T> void writev(std::ostream &f, const std::vector<T> &v) {
  f.write(reinterpret_cast<const char *>(v.data()), sizeof(T) * v.size());
}
struct Fixture {
  I scale, price, base;
  double intensity, p;
  std::vector<Input> components;
  size_t original_events = 0, grouped_events = 0;
  double setup_ms = 0;
  explicit Fixture(const char *path) {
    std::ifstream f(path, std::ios::binary);
    if (read<uint64_t>(f) != 0x31545343)
      throw std::runtime_error("fixture magic");
    const auto n = read<uint64_t>(f);
    scale = read<I>(f);
    intensity = read<double>(f);
    price = read<I>(f);
    p = read<double>(f);
    base = read<I>(f);
    for (size_t j = 0; j < n; ++j) {
      const auto m = read<uint64_t>(f);
      auto real = readv<double>(f, m);
      Input in;
      in.e = readv<I>(f, m);
      in.t = readv<I>(f, m);
      in.theo = readv<uint8_t>(f, m);
      in.grid(real, scale, p, 2 * price);
      original_events += prepare(in, false, false, false, false).events.size();
      grouped_events += prepare(in, true, false, false, false).events.size();
      components.push_back(std::move(in));
    }
  }
};
} // namespace sweep_trials
#endif
