#!/bin/bash

pip uninstall -y wnet
SKBUILD_BUILD_DIR=_skbuild_$(hostname -s) VERBOSE=1 pip install -v -e . --no-build-isolation
