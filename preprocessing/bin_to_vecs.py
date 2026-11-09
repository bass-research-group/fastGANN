"""Convert big-ann-benchmarks binaries to the fvecs / ivecs the benchmarks read.

fbin:  header uint32 [n, d], then n x d float32        -> fvecs, each row int32 d then d float32
gtbin: header uint32 [n, k], then n x k int32 ids, then n x k float32 distances -> ivecs of the ids only

usage: bin_to_vecs.py fbin <in.fbin> <out.fvecs>
       bin_to_vecs.py gtbin <in.bin> <out.ivecs>
"""
import sys

import numpy as np


def to_vecs(rows: np.ndarray, out: str) -> None:
    """Write int32-viewable rows, each prefixed by its width."""
    n, d = rows.shape
    vecs = np.empty((n, d + 1), dtype=np.int32)
    vecs[:, 0] = d
    vecs[:, 1:] = rows.view(np.int32)
    vecs.tofile(out)


def main() -> int:
    if len(sys.argv) != 4 or sys.argv[1] not in ("fbin", "gtbin"):
        print(__doc__, file=sys.stderr)
        return 1
    kind, src, out = sys.argv[1:]
    n, d = (int(x) for x in np.fromfile(src, dtype=np.uint32, count=2))
    dtype = np.float32 if kind == "fbin" else np.int32
    rows = np.fromfile(src, dtype=dtype, count=n * d, offset=8).reshape(n, d)
    to_vecs(rows, out)
    print(f"{out}: {n} x {d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
