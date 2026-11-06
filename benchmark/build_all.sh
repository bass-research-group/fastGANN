#!/bin/bash
# Build all four baselines. Optional argument: CAGRA CUDA architecture (default 89).
set -euo pipefail
B=$(cd "$(dirname "$0")" && pwd)
ARCH=${1:-89}
bash "$B/parlayann/build.sh"
bash "$B/symphonyqg-parallel/build.sh"
bash "$B/jasper/build_jasper.sh"
bash "$B/cagra/build_cuvs.sh" "$ARCH"
bash "$B/cagra/build_bench.sh" "$ARCH"
