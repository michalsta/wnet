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

Simple usage:
```python
import numpy as np
from wnet import WassersteinDistance, Distribution
from wnet.distances import L1Distance

positions1 = np.array(
    [[0, 1, 5, 10],
     [0, 0, 0, 3]]
)
intensities1 = np.array([10, 5, 5, 5])

positions2 = np.array(
    [[1,10],
    [0, 0]])
intensities2 = np.array([20, 5])

S1 = Distribution(positions1, intensities1)
S2 = Distribution(positions2, intensities2)

print(WassersteinDistance(S1, S2, L1Distance()))
# 45
```

## Licence
MIT Licence

## Related Projects

- [pylmcf](https://github.com/cheind/pylmcf) - Python bindings for Min Cost Flow algorithms from LEMON library.
- [LEMON Graph Library](https://lemon.cs.elte.hu/trac/lemon) - C++ library for efficient graph algorithms.