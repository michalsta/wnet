#!/bin/bash
set -e

pip uninstall -y wnet

UBSAN_FLAGS="-fsanitize=undefined -fno-sanitize=vptr -fno-omit-frame-pointer -fno-sanitize-recover=all -Og -g"

CC=clang CXX=clang++ VERBOSE=1 pip install -v -e . \
    --config-settings="cmake.build-type=Debug" \
    --config-settings="cmake.define.CMAKE_CXX_FLAGS=${UBSAN_FLAGS}" \
    --config-settings="cmake.define.CMAKE_SHARED_LINKER_FLAGS=-fsanitize=undefined"

