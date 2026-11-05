# Baselines

Upstream baselines as pinned git submodules, each with a bench program that runs in the paper's configurations.

| baseline | submodule | pinned commit | bench |
|---|---|---|---|
| ParlayANN (CPU) | `parlayann/ParlayANN` (cmuparlay/ParlayANN) | a48c6a3 | `parlayann/bench_parlayann.cc`: ParlayANN's `qsearchAll` on a fixed-degree graph |
| CAGRA (GPU) | `cagra/cuvs` (rapidsai/cuvs) | 0c3d007 (26.08 dev, 2026-06-02) | `cagra/bench_cagra.cu`: cuVS CAGRA on an external graph |
| Jasper (GPU) | `jasper/jasper` (saltsystemslab/jasper) | a0173f4 | Jasper's own `run_query` |

## Build

```bash
git submodule update --init --recursive
cd benchmark/
bash parlayann/build.sh          # g++-12, ParlayANN's flags
bash jasper/build_jasper.sh      # CUDA 13.1, Jasper's CMake
bash cagra/build_cuvs.sh 89      # CUDA 13.1, conda env "cuvs"
bash cagra/build_bench.sh 89
```

cuVS fetches rapids-cmake, raft and rmm from their `main` branches, which no longer build with 0c3d007.
`build_cuvs.sh` pins them to the commits of the paper's build: rapids-cmake 1e88d0d, raft 0903417, rmm 4d354f6.
