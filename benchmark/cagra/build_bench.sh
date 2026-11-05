#!/bin/bash
# Build bench_cagra against the cuvs submodule's build (build_cuvs.sh first) with nvcc, using that build's headers and libraries.
# usage: build_bench.sh [arch=89]   env: CUVS_ENV, CUDA_HOME as build_cuvs.sh
set -euo pipefail
H=$(cd "$(dirname "$0")" && pwd)
ARCH=${1:-89} E=${CUVS_ENV:-$HOME/anaconda3/envs/cuvs} C=${CUDA_HOME:-/usr/local/cuda-13.1}
R=$H/cuvs/cpp BD=$H/build/cuvs
$C/bin/nvcc -std=c++20 -O3 -arch=sm_$ARCH -ccbin $E/bin/x86_64-conda-linux-gnu-c++ \
  -Xcompiler -fopenmp --expt-extended-lambda --expt-relaxed-constexpr \
  -DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE -DRAFT_SYSTEM_LITTLE_ENDIAN=1 \
  -DRAFT_LOG_ACTIVE_LEVEL=RAPIDS_LOGGER_LOG_LEVEL_INFO \
  -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP \
  -I$BD/_deps/cccl-src/thrust -I$BD/_deps/cccl-src/libcudacxx/include -I$BD/_deps/cccl-src/cub \
  -isystem $BD/_deps/rmm-src/cpp/include -isystem $BD/_deps/rmm-build/include \
  -isystem $BD/_deps/rapids_logger-src/include \
  -isystem $H/build/deps/rmm/cpp/include -isystem $H/build/deps/raft/cpp/include \
  -isystem $BD/_deps/raft-build/include -isystem $R/include -isystem $R/src -isystem $BD/include -isystem $E/include \
  -o $H/build/bench_cagra $H/bench_cagra.cu \
  -L$BD -L$BD/_deps/rmm-build -L$BD/_deps/rapids_logger-build -L$E/lib -lcuvs -lrmm -lrapids_logger -lgomp \
  -Xlinker -rpath=$BD:$BD/_deps/rmm-build:$BD/_deps/rapids_logger-build:$E/lib
