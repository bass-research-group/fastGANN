"""Extract the neighbor lists of a SymphonyQG index into the fixed-degree graph file the benchmarks read.

Index file: uint32 entry point, then n rows of float32 [vector (dim) | codes (padded_dim / 64 * 2 * degree) |
factors (3 * degree) | neighbor ids (degree, uint32)], then padded_dim rotator floats; padded_dim is dim rounded up
to a power of two.
Graph file: uint32 [n, degree], then n x degree uint32 neighbor ids.

usage: index_to_graph.py <base.fvecs> <symqg.index> <out.graph.bin> [degree=32]
"""
import os
import sys

import numpy as np


def main() -> int:
    if len(sys.argv) not in (4, 5):
        print(__doc__, file=sys.stderr)
        return 1
    base_path, index_path, out = sys.argv[1:4]
    degree = int(sys.argv[4]) if len(sys.argv) == 5 else 32

    # [1] shape from the base file, row layout from SymphonyQG's QuantizedGraph::initialize
    dim = int(np.fromfile(base_path, dtype=np.int32, count=1)[0])
    n = os.path.getsize(base_path) // (4 * (dim + 1))
    padded = 1 << (dim - 1).bit_length()
    neighbor_offset = dim + padded // 64 * 2 * degree + 3 * degree
    row = neighbor_offset + degree
    expected = 4 + (n * row + padded) * 4
    if os.path.getsize(index_path) != expected:
        print(f"{index_path}: {os.path.getsize(index_path)} bytes, expected {expected} for n={n} dim={dim} "
              f"degree={degree}", file=sys.stderr)
        return 1

    # [2] neighbor ids of every row
    rows = np.memmap(index_path, dtype=np.uint32, mode="r", offset=4, shape=(n, row))
    with open(out, "wb") as f:
        np.array([n, degree], dtype=np.uint32).tofile(f)
        np.ascontiguousarray(rows[:, neighbor_offset:]).tofile(f)
    entry = int(np.fromfile(index_path, dtype=np.uint32, count=1)[0])
    print(f"{out}: {n} x {degree}, entry point {entry}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
