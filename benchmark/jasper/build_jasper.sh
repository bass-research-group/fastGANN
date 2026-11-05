#!/bin/bash
# Build the jasper submodule's run_query (its own CMake, unmodified) as the paper's Jasper runs did: CUDA 13.1, Release,
# command-line tools on, tests and FFI off. run_query's CMake target already lists sm_89/90/120.
# usage: build_jasper.sh   env: CUDA_HOME (/usr/local/cuda-13.1), JOBS (8)
set -euo pipefail
H=$(cd "$(dirname "$0")" && pwd)
C=${CUDA_HOME:-/usr/local/cuda-13.1}
export PATH=$C/bin:$PATH
cmake -S $H/jasper -B $H/build -DCMAKE_BUILD_TYPE=Release -DJASPER_BUILD_CMD=ON -DJASPER_BUILD_TESTS=OFF \
  -DJASPER_BUILD_FFI=OFF -DCMAKE_CUDA_COMPILER=$C/bin/nvcc -DThrust_DIR=$C/lib64/cmake/thrust \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build $H/build -j ${JOBS:-8} --target run_query
