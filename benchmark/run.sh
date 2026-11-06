#!/bin/bash
# Run one fixed beam setting for each MS Turing 1M CPU/GPU baseline.
set -euo pipefail
B=$(cd "$(dirname "$0")" && pwd)
R=$(dirname "$B")
D=$R/dataset/msturing1m
O=${RESULT_DIR:-$R/result/fig01_msturing1m}
PAR=$B/parlayann/build/bench_parlayann
SYMQG=$B/symphonyqg-parallel/build/bench_symqg
CAGRA=$B/cagra/build/bench_cagra
JASPER=$B/jasper/build/bin/run_query
THREADS=${THREADS:-16}
TRIALS=${TRIALS:-3}
[[ $THREADS =~ ^[1-9][0-9]*$ && $TRIALS =~ ^[1-9][0-9]*$ ]] || {
  echo "THREADS and TRIALS must be positive integers" >&2; exit 1; }
for binary in "$PAR" "$SYMQG" "$CAGRA" "$JASPER"; do
  [[ -x $binary ]] || { echo "Missing executable: $binary; run its build script first" >&2; exit 1; }
done
for file in msturing1m_base.fvecs msturing1m_query.fvecs msturing1m_groundtruth.ivecs symqg32.index symqg32.graph.bin symqg32.jasper query100K.fbin msturing-gt-1M; do
  [[ -s $D/$file ]] || { echo "Missing $D/$file; run preprocessing/prepare_msturing1m.sh first" >&2; exit 1; }
done
ENTRY=$("${PYTHON:-python3}" -c 'import struct, sys; print(struct.unpack("<I", open(sys.argv[1], "rb").read(4))[0])' "$D/symqg32.index")
mkdir -p "$O/cpu" "$O/gpu"
export OMP_NUM_THREADS=$THREADS OMP_DYNAMIC=false PARLAY_NUM_THREADS=$THREADS
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
PARLAY_CACHE=$(mktemp -d "${TMPDIR:-/tmp}/fastgann-parlay.XXXXXX")
export PARLAY_CACHE
trap 'rm -rf -- "$PARLAY_CACHE"' EXIT
for ((t=1; t<=TRIALS; t++)); do
  "$PAR" "$D/msturing1m_base.fvecs" "$D/msturing1m_query.fvecs" "$D/msturing1m_groundtruth.ivecs" \
    "$D/symqg32.graph.bin" "$ENTRY" 100 664 1 > "$O/cpu/parlay_b664_t$t.log" 2>&1
  "$SYMQG" -query "$D/msturing1m_query.fvecs" -gt "$D/msturing1m_groundtruth.ivecs" \
    -qg "$D/symqg32.index" -degree 32 -k 100 -beam 1792 -threads "$THREADS" \
    > "$O/cpu/symqg_b1792_t$t.log" 2>&1
  "$CAGRA" "$D/msturing1m_base.fvecs" "$D/msturing1m_query.fvecs" "$D/msturing1m_groundtruth.ivecs" \
    "$D/symqg32.graph.bin" 100 688 multi > "$O/gpu/cagra_b688_t$t.log" 2>&1
  "$JASPER" -i "$D/symqg32.jasper" -q "$D/query100K.fbin" -g "$D/msturing-gt-1M" \
    -t float -d l2 -n 32 --dim 100 -k 100 -b 768:1536 --rabitq_bits 4 \
    > "$O/gpu/jasper_b768_t$t.log" 2>&1
  echo "Finished trial $t/$TRIALS for all four methods"
done
printf '%s\n' "$TRIALS" > "$O/trials.txt"
echo "Logs saved to $O; run benchmark/plotting.sh to generate the figure"
