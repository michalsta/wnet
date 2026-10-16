#!/usr/bin/env bash
# Build and run the standalone C++ suites listed in tests/cpp/suites.txt with
# one toolchain/flag set:
#
#   CXX=clang++ CXXFLAGS="-O1 -g -fsanitize=address,undefined" tests/cpp/run_all.sh
#
# Environment: CXX, CXXFLAGS, LDFLAGS, JOBS, OUT, TIMEOUT, RUN_PREFIX and
# SUITE_LIGHT as documented in .github/scripts/run_cpp_suites.sh. pylmcf's
# headers come from PYLMCF_INCLUDE, else from the installed pylmcf.
set -u
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
PYLMCF_INCLUDE=${PYLMCF_INCLUDE:-$(python3 -m pylmcf --include)}
export CPP_INCLUDES="-I$root/src/wnet/cpp -I$root/src/wnet/cpp/wnet -I$PYLMCF_INCLUDE ${CPP_INCLUDES:-}"
exec "$root/.github/scripts/run_cpp_suites.sh" "$here/suites.txt"
