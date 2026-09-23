"""Plot convergence of both search methods from their saved histories.

Run `simulated_annealing.py` and `genetic_algorithm.py` first; this reads the
`*_history.npz` files they leave in `results/`.

The x axes are deliberately not shared. An annealing iteration is one candidate
evaluation; a GA generation is POP_SIZE of them. Putting both on a single
"steps" axis would imply a cost equivalence that does not exist.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from history import load_histories

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def _load(tag):
    history = os.path.join(RESULTS, f"{tag}_history.npz")
    summary = os.path.join(RESULTS, f"{tag}.json")
    if not (os.path.exists(history) and os.path.exists(summary)):
        return None, None, None
    steps, runs = load_histories(history)
    return steps, runs, json.load(open(summary))


def panel(ax, steps, runs, summary, title, xlabel):
    for run in runs:
        ax.plot(steps, run, alpha=0.35, linewidth=1)
    ax.plot(steps, runs.min(axis=0), linewidth=2.5, color="black", label="best of all runs")
    ax.plot(steps, runs.mean(axis=0), linewidth=2, linestyle="--", color="crimson",
            label="mean over runs")
    ax.set_yscale("symlog")
    ax.set_xscale("log")
    ax.set_title(f"{title}\nbest {summary['score']:,} bytes from budget")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("distance from budget (bytes)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)


def main():
    panels = [
        ("annealing_slack", "Simulated annealing (slack)", "iteration"),
        ("genetic_slack", "Genetic algorithm (slack)", "generation"),
        ("annealing_deviation", "Simulated annealing (deviation)", "iteration"),
        ("genetic_deviation", "Genetic algorithm (deviation)", "generation"),
    ]
    available = [(tag, title, xlabel) for tag, title, xlabel in panels
                 if _load(tag)[1] is not None]
    if not available:
        raise SystemExit("No history files found; run the solvers first.")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (tag, title, xlabel) in zip(axes.ravel(), available):
        steps, runs, summary = _load(tag)
        panel(ax, steps, runs, summary, title, xlabel)
    for ax in axes.ravel()[len(available):]:
        ax.axis("off")

    fig.suptitle("Memory packing: convergence over 20 independent runs", fontsize=13)
    fig.tight_layout()
    out = os.path.join(RESULTS, "convergence.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
