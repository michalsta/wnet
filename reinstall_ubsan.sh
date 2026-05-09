#!/bin/bash
set -e

cd "$(dirname "$0")"

# address:     heap/stack/global buffer overflows, use-after-free
# undefined:   all UB (signed overflow, null deref, bad shift, bounds, ...)
# -fno-sanitize=vptr: skip vtable checks (needs RTTI, causes false positives)
SAN_FLAGS="-fsanitize=address,undefined -fno-sanitize=vptr -fno-sanitize-recover=all -fno-omit-frame-pointer -Og -g"

pip uninstall -y wnet || true
VERBOSE=1 pip install -v -e . \
    --config-settings="cmake.build-type=Debug" \
    --config-settings="cmake.define.CMAKE_CXX_FLAGS=${SAN_FLAGS}"

echo ""
echo "Run tests with:"
echo "  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libasan.so.8 \\"
echo "  PYTHONMALLOC=malloc \\"
echo "  ASAN_OPTIONS=detect_leaks=0:abort_on_error=1:print_stacktrace=1:log_path=/tmp/asan \\"
echo "  UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1:log_path=/tmp/ubsan \\"
echo "  python -m pytest tests/ -v"
