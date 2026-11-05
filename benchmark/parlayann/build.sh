#!/bin/bash
# Build bench_parlayann against the ParlayANN submodule with ParlayANN's own flags (algorithms/bench/parallelDefsANN,
# default HOMEGROWN scheduler). usage: build.sh   env: CXX (default g++-12)
set -euo pipefail
H=$(cd "$(dirname "$0")" && pwd)
mkdir -p $H/build
${CXX:-g++-12} -mcx16 -O3 -std=c++17 -march=native -DNDEBUG -DHOMEGROWN -pthread \
  -I $H/ParlayANN/algorithms -I $H/ParlayANN/parlaylib/include -o $H/build/bench_parlayann $H/bench_parlayann.cc -ldl
