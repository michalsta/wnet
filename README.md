# wnet

Wasserstein Network (wnet) is a Python/C++ library for working with Wasserstein distances. It uses the Min Cost Flow algorithm as implemented by the [LEMON library](https://lemon.cs.elte.hu/trac/lemon), exposed to Python via the [pylmcf module](https://github.com/michalsta/pylmcf), enabling efficient computation and manipulation of Wasserstein distances between multidimensional distributions.

## Features
- Wasserstein and Truncated Wasserstein distance calculations between multidimensional distributions
- Calculation of derivatives with respect to deltas in flow or position (in progress)
- Python and C++ integration
- Support for distribution mixtures, and efficient recalculation of distance with changed mixture proportions

## Installation

You can install the Python package using pip:

```bash
pip install wnet
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

- [pylmcf](https://github.com/michalsta/pylmcf) - Python bindings for Min Cost Flow algorithms from LEMON library.
- [wnetalign](https://github.com/michalsta/wnetalign) - Alignment of MS/NMR spectra using Truncated Wasserstein Distance
