# wnet

Wasserstein Network (wnet) is a Python/C++ library for working with Wasserstein distances. It uses the Min Cost Flow algorithm as implemented by the [LEMON library](https://lemon.cs.elte.hu/trac/lemon), and exposes this functionality to Python via the [pylmcf module](https://github.com/michalsta/pylmcf), enabling efficient computation and manipulation of Wasserstein distances between multidimensional distributions.

## Features
- Wasserstein and Truncated Wasserstein distance calculations
- Calculation of der
- Python and C++ integration

## Installation

You can install the Python package using pip:

```bash
pip install .
```

## Usage

Import the library in Python:

```python
import wnet
```

Example usage:
```python
from wnet import distances, distribution
# Compute Wasserstein distance between distributions
dist = distances.wasserstein_distance(a, b)
```

## License
MIT License

## Related Projects

- [pylmcf](https://github.com/cheind/pylmcf) - Python bindings for Min Cost Flow algorithms from LEMON library.
- [LEMON Graph Library](https://lemon.cs.elte.hu/trac/lemon) - C++ library for efficient graph algorithms.