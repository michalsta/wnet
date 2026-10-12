"""Export existing full NPZ datasets for the native prototype driver."""

import argparse
import json
from pathlib import Path
import struct
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("data", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
meta = json.loads((args.data.parent / "manifest.json").read_text())[args.data.stem]
z = np.load(args.data)
with args.output.open("wb") as out:
    out.write(struct.pack("<Qdd", len(meta["n_theo"]), meta["radius"], 1e8))
    for prefix in ["emp", *[f"theo_{i}" for i in range(len(meta["n_theo"]))]]:
        if prefix == "emp":
            xs, weights = z["emp_pos"].ravel(), z["emp_int"]
        else:
            i = prefix.split("_")[1]
            xs, weights = z[f"theo_pos_{i}"].ravel(), z[f"theo_int_{i}"]
        weights = weights / weights.sum()
        out.write(struct.pack("<Q", len(xs)))
        out.write(np.asarray(xs, dtype="<f8").tobytes())
        out.write(np.asarray(weights, dtype="<f8").tobytes())
