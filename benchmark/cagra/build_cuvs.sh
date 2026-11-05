#!/bin/bash
# Build the cuvs submodule's C++ library for one architecture (default 89) with the conda env's toolchain, as the
# paper's CAGRA runs did: CUDA 13.1, Release, shared libcuvs, no tests / C API / benchmarks / multi-GPU.
# cuVS fetches rapids-cmake, raft and rmm from their main branches; they are pinned here to the commits of the
# paper's 2026-06-02 build (rapids-cmake by sha, raft and rmm as local clones handed to CPM).
# usage: build_cuvs.sh [arch=89]   env: CUVS_ENV (conda env with cuVS's build deps), CUDA_HOME (/usr/local/cuda-13.1)
set -euo pipefail
H=$(cd "$(dirname "$0")" && pwd)
ARCH=${1:-89} E=${CUVS_ENV:-$HOME/anaconda3/envs/cuvs} C=${CUDA_HOME:-/usr/local/cuda-13.1}
RAPIDS_CMAKE=1e88d0d60c1ad1dc2d55cdf5e1ff2f27806e97f6
RAFT=0903417414a6b77aaa1086a48aa9d6bd5763b06a
RMM=4d354f6ba28327ba32a875e099461e8048c76c26
D=$H/build/deps
pin() {  # name url sha
  [[ -d $D/$1 ]] || git clone -q $2 $D/$1
  git -C $D/$1 fetch -q origin $3
  git -C $D/$1 checkout -q $3
}
mkdir -p $D
pin raft https://github.com/rapidsai/raft.git $RAFT
pin rmm https://github.com/rapidsai/rmm.git $RMM
export PATH=$E/bin:$PATH CONDA_PREFIX=$E
cmake -S $H/cuvs/cpp -B $H/build/cuvs -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=$C/bin/nvcc \
  -DCMAKE_CXX_COMPILER=$E/bin/x86_64-conda-linux-gnu-c++ -DCMAKE_CUDA_ARCHITECTURES=$ARCH-real \
  -Drapids-cmake-sha=$RAPIDS_CMAKE -DCPM_raft_SOURCE=$D/raft -DCPM_rmm_SOURCE=$D/rmm \
  -DCMAKE_INSTALL_PREFIX=$H/build/install -DBUILD_SHARED_LIBS=ON -DBUILD_TESTS=OFF -DBUILD_C_LIBRARY=OFF \
  -DBUILD_CUVS_BENCH=OFF -DBUILD_MG_ALGOS=OFF -DBUILD_CAGRA_HNSWLIB=OFF -DCUVS_NVTX=OFF \
  -DCUVS_COMPILE_DYNAMIC_ONLY=ON -DDISABLE_DEPRECATION_WARNINGS=ON
cmake --build $H/build/cuvs -j ${JOBS:-16}
