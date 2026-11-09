"""Build an upstream SymphonyQG index (L2, FastScan codebook) from base vectors with its Python bindings.

usage: build_symqg_index.py <base.fvecs> <out.index> [degree=32] [ef=400] [iters=3]
"""
import sys
import time

import numpy as np
import symphonyqg


def read_fvecs(path: str) -> np.ndarray:
    """Rows of an fvecs file as a contiguous float32 matrix."""
    raw = np.fromfile(path, dtype=np.int32)
    dim = int(raw[0])
    return np.ascontiguousarray(raw.reshape(-1, dim + 1)[:, 1:].view(np.float32))


def main() -> int:
    if not 3 <= len(sys.argv) <= 6:
        print(__doc__, file=sys.stderr)
        return 1
    base_path, out = sys.argv[1], sys.argv[2]
    degree, ef, iters = (int(x) for x in (sys.argv[3:] + ["32", "400", "3"][len(sys.argv) - 3:]))
    base = read_fvecs(base_path)
    n, dim = base.shape
    index = symphonyqg.Index("QG", "L2", num_elements=n, dimension=dim, degree_bound=degree)
    start = time.time()
    index.build_index(base, ef, num_iter=iters)
    print(f"{out}: {n} x {dim}, degree {degree}, EF {ef}, {iters} iterations, {time.time() - start:.0f} s")
    index.save(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
