#!/bin/bash
# Collect one median-QPS point per method and render Figure 1's QPS panel.
set -euo pipefail
B=$(cd "$(dirname "$0")" && pwd)
R=$(dirname "$B")
O=${RESULT_DIR:-$R/result/fig01_msturing1m}
if [[ -z ${PYTHON:-} ]]; then
  if [[ -x $R/.venv/bin/python ]]; then PYTHON=$R/.venv/bin/python; else PYTHON=python3; fi
fi
if [[ -z ${TRIALS:-} ]]; then
  [[ -f $O/trials.txt ]] || { echo "Run benchmark/run.sh first, or set TRIALS for existing logs" >&2; exit 1; }
  TRIALS=$(cat "$O/trials.txt")
fi
[[ $TRIALS =~ ^[1-9][0-9]*$ ]] || { echo "TRIALS must be a positive integer" >&2; exit 1; }
"$PYTHON" - "$O" "$TRIALS" <<'PYTHON'
import csv
import math
import re
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

METHODS = {
    "cpu": [("Baseline", "ParlayANN", "parlay", 664),
            ("Quantized", "SymphonyQG parallel", "symqg", 1792)],
    "gpu": [("Baseline", "CAGRA", "cagra", 688),
            ("Quantized", "Jasper", "jasper", 768)],
}


def one_match(pattern: str, text: str):
    matches = re.findall(pattern, text, re.MULTILINE)
    if len(matches) != 1:
        raise ValueError(f"Expected one result matching {pattern!r}, found {len(matches)}")
    return matches[0]


def point(method: str, beam: int, text: str) -> tuple[float, float]:
    if method == "parlay":
        row = next(csv.reader([one_match(rf"^({beam},1,[^\n]+)$", text)]))
        recall, qps = float(row[2]), float(row[3])
    elif method == "symqg":
        recall = float(one_match(r"runtime: [^\n]+recall@100: ([0-9.eE+-]+)", text))
        qps = float(one_match(r"throughput: ([0-9.eE+-]+) queries/sec", text))
    elif method == "cagra":
        row = next(csv.reader([one_match(rf"^(100,{beam},[^\n]+)$", text)]))
        qps, recall = float(row[4]), float(row[5])
    else:
        qps = float(one_match(rf"beam_width={beam} limit=1536 [^\n]*throughput=([0-9.eE+-]+) QPS", text))
        recall = 100 * float(one_match(r"Recall@100: ([0-9.eE+-]+)", text))
    if not math.isfinite(qps) or qps <= 0 or not 0 <= recall <= 100:
        raise ValueError(f"Invalid result: QPS={qps}, recall={recall}")
    return qps, recall


def collect(platform: str, directory: Path, trials: int) -> list[dict]:
    rows = []
    for family, system, method, beam in METHODS[platform]:
        values = [point(method, beam, (directory / f"{method}_b{beam}_t{t}.log").read_text())
                  for t in range(1, trials + 1)]
        rows.append({"platform": platform, "implementation": family, "system": system,
                     "beam": beam, "recall": statistics.median(v[1] for v in values),
                     "qps": statistics.median(v[0] for v in values), "qps_trials": trials})
    return rows


RED = "#d64545"
BLUE, BLUE_EDGE = "#9db9dc", "#3269b7"
GROUPS = [("cpu", "CPU"), ("gpu", "GPU")]
FAMILIES = [("Baseline", RED, RED, None), ("Quantized", BLUE, BLUE_EDGE, "///")]
BAR_W, GAP = 0.8, 1.0


def main():
    directory = Path(sys.argv[1]).resolve()
    trials = int(sys.argv[2])
    rows = collect("cpu", directory / "cpu", trials) + collect("gpu", directory / "gpu", trials)
    by_key = {(r["platform"], r["implementation"]): r for r in rows}
    expected = {(p, family) for p, _ in GROUPS for family, *_ in FAMILIES}
    if len(rows) != len(expected) or set(by_key) != expected:
        raise ValueError("Expected one summary row for each of ParlayANN, SymphonyQG, CAGRA, and Jasper")

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                         "font.size": 11.5, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11,
                         "legend.fontsize": 11, "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "hatch.linewidth": 0.8})
    fig, ax = plt.subplots(figsize=(252 / 72, 182.4 / 72))
    fig.subplots_adjust(left=0.20, right=0.98, top=0.97, bottom=0.19)
    centers, x = [], 0.0
    for platform, _ in GROUPS:
        xs = []
        for family, fill, edge, hatch in FAMILIES:
            height = float(by_key[(platform, family)]["qps"]) / 1e3
            ax.bar(x, height, BAR_W, color=fill, hatch=hatch, edgecolor=edge, lw=0)
            ax.bar(x, height, BAR_W, fill=False, edgecolor="black", lw=0.7)
            xs.append(x)
            x += 1
        centers.append(sum(xs) / len(xs))
        x += GAP
    xmax = x - GAP - 1
    ymax = max(70, math.ceil(max(float(r["qps"]) for r in rows) / 10000) * 10)
    ax.set_xlim(-0.75, xmax + 0.75)
    ax.set_ylim(0, ymax)
    ax.set_yticks(range(0, ymax + 1, 10))
    ax.set_xticks(centers)
    ax.set_xticklabels([label for _, label in GROUPS], fontsize=13)
    ax.tick_params(axis="x", length=0, pad=6)
    ax.set_ylabel("Queries/sec (10³)")
    ax.yaxis.grid(True, color="#dddddd", lw=0.6)
    ax.set_axisbelow(True)
    sep = len(FAMILIES) - 1 + (GAP + 1) / 2
    ax.plot([sep, sep], [-0.02, -0.16], transform=ax.get_xaxis_transform(), color="#999999", lw=0.8, clip_on=False)
    handles = [Patch(facecolor=RED, edgecolor="black", lw=0.7),
               Patch(facecolor=BLUE, edgecolor=BLUE_EDGE, hatch="///", lw=0.7)]
    ax.legend(handles, ["ParlayANN / CAGRA", "SymphonyQG / Jasper"], loc="upper left", frameon=False,
              handlelength=1.6, handleheight=1.0, labelspacing=0.3, borderaxespad=0.4)
    for suffix in ("pdf", "png"):
        path = directory / f"fig01.{suffix}"
        fig.savefig(path, dpi=250)
        print(path)
    plt.close(fig)
    with (directory / "plot_data.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


main()
PYTHON
