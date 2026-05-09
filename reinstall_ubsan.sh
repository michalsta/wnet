#!/bin/bash
set -e

cd "$(dirname "$0")"

# address:     heap/stack/global buffer overflows, use-after-free
# undefined:   all UB (signed overflow, null deref, bad shift, bounds, ...)
# -fno-sanitize=vptr: skip vtable checks (needs RTTI, causes false positives)
# -shared-libasan: link libasan as a dynamic .so dependency instead of
#   embedding it statically. This means libasan is loaded when Python imports
#   wnet (not at Python startup via LD_PRELOAD), so libstdc++ is already
#   in memory and ASan can intercept __cxa_throw correctly.
SAN_FLAGS="-fsanitize=address,undefined -fno-sanitize=vptr -fno-sanitize-recover=all -fno-omit-frame-pointer -Og -g"

pip uninstall -y wnet || true
VERBOSE=1 pip install -v -e . \
    --config-settings="cmake.build-type=Debug" \
    --config-settings="cmake.define.CMAKE_CXX_FLAGS=${SAN_FLAGS}"

echo ""
echo "Run tests with:"
echo "  PYTHONMALLOC=malloc \\"
echo "  ASAN_OPTIONS=detect_leaks=0:abort_on_error=1:print_stacktrace=1:log_path=/tmp/asan:verify_asan_link_order=0 \\"
echo "  UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1:log_path=/tmp/ubsan \\"
echo "  python -m pytest tests/ -v"
echo ""
echo "Then view reports with:"
echo "  cat /tmp/asan.* 2>/dev/null"
echo "  cat /tmp/ubsan.* 2>/dev/null"
