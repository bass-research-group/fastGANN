#!/bin/bash
# Download big-ann-benchmarks' MS Turing 1M subset into dataset/msturing1m/ and convert it to fvecs / ivecs.
# Each step skips outputs that already exist.
# usage: prepare_msturing1m.sh   env: PYTHON (python3, with preprocessing/requirements.txt installed)
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
