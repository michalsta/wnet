"""Assert exact native profiles, grouped optima, certificates and sensitivities.

These tests compile the production headers directly, independently of whichever
extension is installed. The normal Python suites exercise dispatch and bindings.
"""
import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("suite", ["profiles", "grouped"])
def test_native_convex_sweep(suite, tmp_path):
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        pytest.skip("native ConvexSweep checks require a C++ compiler")
    root = Path(__file__).resolve().parents[1]
    exe = tmp_path / suite
    subprocess.run(
        [compiler, "-std=c++20", "-O2", "-pthread", f"-I{root / 'src/wnet/cpp'}",
         f"-I{root / 'src/wnet/cpp/wnet'}",
         str(root / f"tests/cpp/test_convex_sweep_{suite}.cpp"), "-o", str(exe)],
        check=True, timeout=300,
    )
    subprocess.run([str(exe), "3000"], check=True, timeout=180)
