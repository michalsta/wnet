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

# WNET_NB_LINKED=ON: force the classic linked nanobind build. The published
# split-mode backend is an uninstrumented libstdc++ binary, and split mode
# requires the frontend and backend to agree on compiler, C++ runtime and debug
# mode -- neither holds under ASan/UBSan.

# pylmcf has to come along. wnet compiles its headers in and the two share
# nanobind type registrations at runtime, so a stock split-mode pylmcf wheel
# next to a linked wnet fails at import ("nanobind build modes disagree"), and
# an uninstrumented pylmcf would be a blind spot in the middle of the run.
# Rebuild it here with the same flags and the same mode. Point
# PYLMCF_SRC at a checkout to build that instead of the PyPI sdist.
PYLMCF_SRC="${PYLMCF_SRC:-}"
pip uninstall -y pylmcf wnet || true
if [ -n "$PYLMCF_SRC" ]; then
    PYLMCF_TARGET=("$PYLMCF_SRC")
else
    PYLMCF_TARGET=(--no-binary pylmcf pylmcf)
fi
VERBOSE=1 pip install -v "${PYLMCF_TARGET[@]}" \
    --config-settings="cmake.build-type=Debug" \
    --config-settings="cmake.define.WNET_NB_LINKED=ON" \
    --config-settings="cmake.define.CMAKE_CXX_FLAGS=${SAN_FLAGS}"

VERBOSE=1 pip install -v -e . \
    --config-settings="cmake.build-type=Debug" \
    --config-settings="cmake.define.WNET_NB_LINKED=ON" \
    --config-settings="cmake.define.CMAKE_CXX_FLAGS=${SAN_FLAGS}"

echo ""
echo "This venv now holds instrumented, linked-mode pylmcf and wnet builds."
echo "Restore the normal split-mode stack with ../reinstall_all.sh (or"
echo "pylmcf/reinstall.sh followed by ./reinstall.sh)."
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
