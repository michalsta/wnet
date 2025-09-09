#!/bin/bash

pip uninstall -y wnet
VERBOSE=1 pip install -v -e .
