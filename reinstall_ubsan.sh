#!/bin/bash
set -e

cd "$(dirname "$0")"

UBSAN_FLAGS="-fsanitize=undefined -fno-sanitize=vptr -fno-omit-frame-pointer -fno-sanitize-recover=all -Og -g"
UBSAN_RT=$(find /usr/lib/llvm-* /usr/lib/clang -name "libclang_rt.ubsan_standalone-$(uname -m).so" 2>/dev/null | head -1)

if [[ -z "$UBSAN_RT" ]]; then
    echo "ERROR: could not find libclang_rt.ubsan_standalone shared library" >&2
    exit 1
fi
echo "UBSan runtime: $UBSAN_RT"

pip uninstall -y wnet

CC=clang CXX=clang++ VERBOSE=1 pip install -v -e . \
    --config-settings="cmake.build-type=Debug" \
    --config-settings="cmake.define.CMAKE_CXX_FLAGS=${UBSAN_FLAGS}"

echo ""
echo "Run tests with:"
echo "  LD_PRELOAD=${UBSAN_RT} UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1 python -m pytest tests/ -v"

