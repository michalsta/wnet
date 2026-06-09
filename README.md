# wnet

Wasserstein Network (wnet) is a Python/C++ library for working with Wasserstein distances. It uses the Min Cost Flow algorithm as implemented by the [LEMON library](https://lemon.cs.elte.hu/trac/lemon), exposed to Python via the [pylmcf module](https://github.com/michalsta/pylmcf), enabling efficient computation and manipulation of Wasserstein distances between multidimensional distributions.

## Features
- Wasserstein and Truncated Wasserstein distance between multidimensional distributions (dimensions 1–20)
- Three distance metrics: L1, L2, L∞
- Order-p (Lp) Wasserstein for any real p ≥ 1 (per-pair cost = `ground_distance**p`, result is the p-th root; fractional p via automatic cost scaling)
- Derivatives with respect to peak intensities and spectrum mixture proportions
- Position gradients (∂cost/∂position) with warm-restart re-solving after peak position updates
- Support for distribution mixtures and efficient recalculation with changed mixture proportions
- Picklable `Distribution` objects

## Installation

You can install the Python package using pip:

```bash
pip install wnet
```

## Usage

### Basic distance

```python
import numpy as np
from wnet import WassersteinDistance, Distribution
from wnet.distances import DistanceMetric

positions1 = np.array([[0, 1, 5, 10], [0, 0, 0, 3]])
intensities1 = np.array([10, 5, 5, 5])

positions2 = np.array([[1, 10], [0, 0]])
intensities2 = np.array([20, 5])

S1 = Distribution(positions1, intensities1)
S2 = Distribution(positions2, intensities2)

print(WassersteinDistance(S1, S2, DistanceMetric.L1))
# 45
```

### Order-p (Lp) Wasserstein

By default the distance is the 1-Wasserstein distance. Pass `p` to use the order-p
Wasserstein distance, where each unit of mass moved a ground distance `d` costs `d**p`
and the returned value is the p-th root of the optimal transport cost:

```python
# Quadratic (W2) Wasserstein distance with a Euclidean ground metric
print(WassersteinDistance(S1, S2, DistanceMetric.L2, p=2))
print(TruncatedWassersteinDistance(S1, S2, DistanceMetric.L2, max_distance=3.0, p=2))

# Fractional orders work too (e.g. p = 1.5)
print(WassersteinDistance(S1, S2, DistanceMetric.L2, p=1.5))
```

`p` can be any real number ≥ 1. The ground metric (L1/L2/L∞) is chosen independently of `p`.
For `p != 1` the dense transport network is always used — the fast 1D chain solver is only
valid for `p == 1`, since exponentiated step costs are not additive along a chain.

`p == 1` is bit-exact with the classic 1-Wasserstein distance. For `p != 1` the cost
`d**p` is fractional, so the integer min-cost-flow solver works in automatically scaled
units (`round(scale_factor() * d**p)`); the public results divide that scale back out, so
no tuning is needed.

> At the `WassersteinNetwork` level, `total_cost()` and all derivatives are in `W_p**p`
> units (the sum of `d**p · flow`); take the p-th root for the literal `W_p` distance,
> as `WassersteinDistance()` does.

### Truncated Wasserstein

Mass that cannot be matched within `max_distance` is discarded at a fixed cost rather than transported arbitrarily far:

```python
from wnet import TruncatedWassersteinDistance

print(TruncatedWassersteinDistance(S1, S2, DistanceMetric.L2, max_distance=3.0))
```

### Derivatives w.r.t. peak intensities

`signal_part_derivatives()` returns the marginal cost of increasing each theoretical peak's intensity by 1 — useful for scoring how well each peak is explained:

```python
from wnet import WassersteinNetwork

W = WassersteinNetwork(S1, [S2], DistanceMetric.L2, max_distance=10.0)
W.build()
W.solve()

derivs = W.signal_part_derivatives()   # np.ndarray, one value per peak in S1
```

### Optimising peak positions

After an initial solve, positions can be updated and re-solved cheaply via a warm restart. `update_positions_and_get_gradient()` returns `∂cost/∂position` for all peaks so you can feed them directly into a gradient-based optimiser:

```python
W = WassersteinNetwork(S1, [S2], DistanceMetric.L2, max_distance=10.0)
W.build()
W.solve()

for _ in range(100):
    grad_empirical, grad_theoretical = W.update_positions_and_get_gradient(new_positions)
    new_positions -= 0.01 * grad_empirical
```

## Licence
MIT Licence

## Citation

If you use this software, please cite:

Król J, Bochenek M, Jopa S, Kazimierczuk K, Gambin A, Startek MP (2026).
WNetAlign: fast and accurate spectra alignment using truncated Wasserstein distance and network simplex.
*Briefings in Bioinformatics*, 27(3), bbag247.
https://doi.org/10.1093/bib/bbag247

```bibtex
@article{krol2026wnetalign,
  title   = {WNetAlign: fast and accurate spectra alignment using truncated Wasserstein distance and network simplex},
  author  = {Kr{\'o}l, Justyna and Bochenek, Maria and Jopa, Sylwia and Kazimierczuk, Krzysztof and Gambin, Anna and Startek, Micha{\l} Piotr},
  journal = {Briefings in Bioinformatics},
  volume  = {27},
  number  = {3},
  pages   = {bbag247},
  year    = {2026},
  doi     = {10.1093/bib/bbag247}
}
```

## Related Projects

- [pylmcf](https://github.com/michalsta/pylmcf) - Python bindings for Min Cost Flow algorithms from LEMON library.
- [wnetalign](https://github.com/michalsta/wnetalign) - Alignment of MS/NMR spectra using Truncated Wasserstein Distance
