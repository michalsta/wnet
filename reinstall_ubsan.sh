#!/bin/bash
set -e

cd "$(dirname "$0")"

ASAN_FLAGS="-fsanitize=address -fno-sanitize-recover=all -fno-omit-frame-pointer -Og -g"

pip uninstall -y wnet || true
VERBOSE=1 pip install -v -e . \
    --config-settings="cmake.build-type=Debug" \
    --config-settings="cmake.define.CMAKE_CXX_FLAGS=${ASAN_FLAGS}"

echo ""
echo "Run tests with:"
echo "  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libasan.so.8 ASAN_OPTIONS=detect_leaks=0:abort_on_error=1:print_stacktrace=1 python -m pytest tests/ -v"
