"""Export a fixed-degree graph to the pinned Jasper submodule's native format.

See include/jasper/index/graph.cuh: save_graph_to_file / load_graph_from_file.
Records contain FP16 vectors, uint8 degree, uint32 edges, BF16 edge distances
(unused by search, set to zero), and uint32 identity slot IDs. The file ends
with a zero deletion mask and the next ID. The codebook supplies the entry point.

usage: graph_to_jasper.py <base.fvecs> <graph.bin> <codebook.index> <out.jasper>
"""
import sys

import numpy as np


def main() -> int:
    if len(sys.argv) != 5:
        print(__doc__, file=sys.stderr)
        return 1
    base_path, graph_path, index_path, out = sys.argv[1:]
    raw = np.fromfile(base_path, dtype=np.int32)
    dim = int(raw[0])
    base = raw.reshape(-1, dim + 1)[:, 1:].view(np.float32)
    n, degree = (int(x) for x in np.fromfile(graph_path, dtype=np.uint32, count=2))
    ids = np.fromfile(graph_path, dtype=np.uint32, offset=8).reshape(n, degree)
    entry = int(np.fromfile(index_path, dtype=np.uint32, count=1)[0])
    if n != len(base) or not 0 < degree <= 255 or entry >= n or np.any(ids >= n):
        raise ValueError("Base vectors, graph, degree, or entry point do not match")

    records = np.zeros(n, dtype=np.dtype([
        ("vec", "<f2", (dim,)), ("count", "u1"), ("edges", "<u4", (degree,)),
        ("dist", "<u2", (degree,)), ("slot", "<u4"),
    ], align=False))
    records["vec"] = base.astype(np.float16)
    records["count"] = degree
    records["edges"] = ids
    records["slot"] = np.arange(n, dtype=np.uint32)
    size = 32 + records.nbytes + (n + 7) // 8 + 8
    with open(out, "wb") as f:
        np.array([size, n, entry, records.dtype.itemsize], dtype="<u8").tofile(f)
        records.tofile(f)
        np.zeros((n + 7) // 8, dtype=np.uint8).tofile(f)
        np.array([n], dtype="<u8").tofile(f)
    print(f"{out}: n={n}, dim={dim}, degree={degree}, entry={entry}, bytes={size}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
