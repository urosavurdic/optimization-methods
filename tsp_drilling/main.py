"""Compare exhaustive search against 2-opt on the drilling-path problem.

Exhaustive search is run only where it is tractable. At 20 holes there are
20!/2 ~ 1.2e18 distinct orderings, so the heuristic is all that remains.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from brute_force import brute_force_tsp
from geometry import load_points, path_length
from two_opt import two_opt_multistart

HERE = os.path.dirname(os.path.abspath(__file__))
COORDS = os.path.join(HERE, "coordinates.json")
RESULTS = os.path.join(HERE, "results")

EXACT_LIMIT = 12  # 12 holes ~ 3 minutes; 13 would be ~40


def plot_tours(tours, filename):
    """Draw each solved instance as holes plus the path through them."""
    fig, axes = plt.subplots(1, len(tours), figsize=(5 * len(tours), 5))
    if len(tours) == 1:
        axes = [axes]

    for ax, (n, entry) in zip(axes, sorted(tours.items(), key=lambda kv: int(kv[0]))):
        points = load_points(COORDS, num_points=int(n))
        path = entry["best"]["path"]
        xs = [points[i][0] for i in path]
        ys = [points[i][1] for i in path]

        ax.plot(xs, ys, "-o", linewidth=1.6, markersize=6, zorder=2)
        ax.plot(xs[0], ys[0], "s", markersize=11, fillstyle="none",
                markeredgewidth=1.8, zorder=3, label="start")
        ax.plot(xs[-1], ys[-1], "D", markersize=9, fillstyle="none",
                markeredgewidth=1.8, zorder=3, label="end")
        for order, idx in enumerate(path):
            ax.annotate(str(order + 1), points[idx], textcoords="offset points",
                        xytext=(6, 5), fontsize=8, alpha=0.75)

        ax.set_title(f"{n} holes - {entry['best']['length']:.2f} mm")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(loc="best", fontsize=8)

    fig.suptitle("Shortest drilling path (open, no return to start)")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", type=int, nargs="+", default=[8, 12, 20])
    ap.add_argument("--restarts", type=int, default=50)
    ap.add_argument("--exact-limit", type=int, default=EXACT_LIMIT)
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    tours = {}

    for n in args.sizes:
        points = load_points(COORDS, num_points=n)
        print(f"\n=== {n} holes ===")

        entry = {}
        heuristic = two_opt_multistart(points, restarts=args.restarts)
        print(f"2-opt          {heuristic['length']:8.2f}  "
              f"{heuristic['restarts']} restarts, {heuristic['time']:.3f}s")
        entry["two_opt"] = {k: (list(v) if k == "path" else v)
                            for k, v in heuristic.items()}

        if n <= args.exact_limit:
            exact = brute_force_tsp(points, progress_every=0)
            print(f"exhaustive     {exact['length']:8.2f}  "
                  f"{exact['permutations_checked']:,} orderings, {exact['time']:.1f}s")
            entry["exhaustive"] = {"path": list(exact["path"]),
                                   "length": exact["length"],
                                   "orderings_checked": exact["permutations_checked"],
                                   "time": exact["time"]}
            gap = heuristic["length"] / exact["length"] - 1.0
            if abs(gap) < 1e-12:      # same tour; the residue is float noise
                gap = 0.0
            entry["gap_to_optimal"] = gap
            print(f"gap            {gap:8.4%}")
            entry["best"] = entry["exhaustive"]
        else:
            print(f"exhaustive        n/a  ({n}!/2 orderings is not enumerable)")
            entry["best"] = entry["two_opt"]

        tours[str(n)] = entry

    with open(os.path.join(RESULTS, "tours.json"), "w") as f:
        json.dump(tours, f, indent=2)

    plot_tours(tours, os.path.join(RESULTS, "tour_plot.png"))

    lines = [
        "# Drilling path: exhaustive search vs 2-opt",
        "",
        "Open path, L2 distances, coordinates in millimetres.",
        "",
        "| holes | exhaustive | orderings | time | 2-opt | restarts | time | gap |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for n in sorted(tours, key=int):
        e = tours[n]
        h = e["two_opt"]
        if "exhaustive" in e:
            x = e["exhaustive"]
            lines.append(
                f"| {n} | {x['length']:.2f} | {x['orderings_checked']:,} | {x['time']:.2f}s "
                f"| {h['length']:.2f} | {h['restarts']} | {h['time']:.3f}s "
                f"| {e['gap_to_optimal']:.2%} |")
        else:
            lines.append(
                f"| {n} | not enumerable | {'~1.2e18' if int(n) == 20 else '-'} | - "
                f"| {h['length']:.2f} | {h['restarts']} | {h['time']:.3f}s | - |")
    lines.append("")
    with open(os.path.join(RESULTS, "benchmark.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nWrote {RESULTS}")


if __name__ == "__main__":
    main()
