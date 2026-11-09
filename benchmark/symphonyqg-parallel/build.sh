#!/bin/bash
# Build the parallel CPU SymphonyQG benchmark. Requires an AVX-512 CPU and OpenMP.
set -euo pipefail
H=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$H/build"
${CXX:-g++-12} -std=c++17 -Ofast -march=native -fopenmp -pthread -o "$H/build/bench_symqg" "$H/symqg_parallel.cpp"
