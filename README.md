# fastGANN

fastGANN is an accuracy- and hardware-aware graph-based approximate nearest neighbor search (ANNS) system designed for efficient search on GPUs.

🚧 This project is under active development. The current branch includes ParlayANN, CAGRA, and Jasper baselines, benchmarking scripts, and dataset preparation tools.

## Baselines

Upstream baselines as pinned git submodules, each with a bench program that runs in the paper's configurations.

| baseline | submodule | pinned commit | bench |
|---|---|---|---|
| ParlayANN (CPU) | `benchmark/parlayann/ParlayANN` (cmuparlay/ParlayANN) | a48c6a3 | `benchmark/parlayann/bench_parlayann.cc`: ParlayANN's `qsearchAll` on a fixed-degree graph |
| CAGRA (GPU) | `benchmark/cagra/cuvs` (rapidsai/cuvs) | 0c3d007 (26.08 dev, 2026-06-02) | `benchmark/cagra/bench_cagra.cu`: cuVS CAGRA on an external graph |
| Jasper (GPU) | `benchmark/jasper/jasper` (saltsystemslab/jasper) | a0173f4 | Jasper's own `run_query` |

### Build

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

## Dataset Preparation

`preprocessing/prepare_msturing1m.sh` downloads big-ann-benchmarks' MS Turing 1M subset (about 520 MB) into
`dataset/msturing1m/` and converts it to benchmark inputs:

| file | contents |
|---|---|
| `msturing1m_base.fvecs` | 1M base vectors, 100 dimensions |
| `msturing1m_query.fvecs` | 100K query vectors, 100 dimensions |
| `msturing1m_groundtruth.ivecs` | top-100 ground-truth ids for each query |

Requires Python 3.11 or newer. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r preprocessing/requirements.txt
bash preprocessing/prepare_msturing1m.sh
```

Existing download and conversion outputs are reused. Generated data and the virtual environment are git-ignored.
