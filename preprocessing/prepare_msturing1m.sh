#!/bin/bash
# Download big-ann-benchmarks' MS Turing 1M subset (the first 1M vectors of base1b.fbin, query100K.fbin,
# msturing-gt-1M) into dataset/msturing1m/, convert it to the fvecs / ivecs the benchmarks read, build the upstream
# SymphonyQG index (FastScan codebook) on it and extract that index's graph. Each step skips outputs that already exist.
# usage: prepare_msturing1m.sh   env: PYTHON (python3, with preprocessing/requirements.txt and the symphonyqg bindings)
set -euo pipefail
P=$(cd "$(dirname "$0")" && pwd)
O=$(dirname "$P")/dataset/msturing1m
PY=${PYTHON:-python3}
mkdir -p "$O"

# [1] download through big-ann-benchmarks, its private-query files off
(cd "$P/big-ann-benchmarks" && "$PY" - "$O" <<'EOF'
import sys
from benchmark.datasets import DATASETS
ds = DATASETS["msturing-1M"]()
ds.basedir = sys.argv[1]
ds.private_qs_url = ds.private_gt_url = None
ds.prepare()
EOF
)

# [2] convert to fvecs / ivecs
convert() {  # kind source output
  [[ -s $O/$3 ]] || "$PY" "$P/bin_to_vecs.py" "$1" "$O/$2" "$O/$3"
}
convert fbin base1b.fbin.crop_nb_1000000 msturing1m_base.fvecs
convert fbin query100K.fbin msturing1m_query.fvecs
convert gtbin msturing-gt-1M msturing1m_groundtruth.ivecs

# [3] SymphonyQG index: degree 32, EF 400, 3 iterations
if [[ ! -s $O/symqg32.index ]]; then
  "$PY" -c "import symphonyqg" 2>/dev/null || {
    echo "symphonyqg missing: pip install --no-build-isolation --no-deps preprocessing/SymphonyQG/python" >&2; exit 1; }
  "$PY" "$P/build_symqg_index.py" "$O/msturing1m_base.fvecs" "$O/symqg32.index" 32 400 3
fi

# [4] the index's graph, read by ParlayANN, CAGRA and Jasper
[[ -s $O/symqg32.graph.bin ]] || "$PY" "$P/index_to_graph.py" "$O/msturing1m_base.fvecs" "$O/symqg32.index" \
  "$O/symqg32.graph.bin" 32
