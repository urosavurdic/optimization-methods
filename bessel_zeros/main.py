"""Locate zeros of the spherical Bessel functions by bisection.

Bisection is the slowest root-finder in common use -- one bit of precision per
iteration -- but it is the only one that cannot diverge: given a sign change it
is guaranteed to converge. That makes it the right tool when the job is to find
*every* root in a range rather than to polish one quickly.

Two stages: scan a dense grid for sign changes to bracket each root, then
bisect inside each bracket to full double precision.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import spherical_jn

TOLERANCE = 1e-12
MAX_ITER = 100
X_MAX = 20.0
GRID = 5000

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def bisection(f, a, b, tolerance=TOLERANCE, max_iter=MAX_ITER):
    """Find a root in [a, b], assuming f(a) and f(b) have opposite signs."""
    fa, fb = f(a), f(b)
    if fa * fb >= 0:
        return None

    for _ in range(max_iter):
        midpoint = 0.5 * (a + b)
        if b - a < tolerance:
            return midpoint
        if fa * f(midpoint) < 0:
            b = midpoint
        else:
            a, fa = midpoint, f(midpoint)

    return 0.5 * (a + b)


def find_zeros(order, x_max=X_MAX, grid=GRID):
    """All zeros of j_order on (0, x_max], in ascending order."""
    f = lambda t: spherical_jn(order, t)
    xs = np.linspace(0.01, x_max, grid)
    values = f(xs)

    zeros = []
    for i in range(len(xs) - 1):
        if values[i] * values[i + 1] < 0:
            root = bisection(f, xs[i], xs[i + 1])
            if root is not None:
                zeros.append(root)
    return zeros


def plot(zeros, filename, x_max=X_MAX):
    x = np.linspace(0.01, x_max, 2000)
    fig, ax = plt.subplots(figsize=(9, 5))
    for order in sorted(zeros):
        line, = ax.plot(x, spherical_jn(order, x), label=f"$j_{order}(x)$")
        ax.plot(zeros[order], np.zeros(len(zeros[order])), "o",
                color=line.get_color(), markersize=5)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("$j_n(x)$")
    ax.set_title("Spherical Bessel functions and their zeros")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--orders", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--x-max", type=float, default=X_MAX)
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    zeros = {order: find_zeros(order, x_max=args.x_max) for order in args.orders}

    payload = {}
    for order, roots in sorted(zeros.items()):
        print(f"Zeros of j_{order}(x) on (0, {args.x_max}]:")
        entries = []
        for k, root in enumerate(roots, start=1):
            residual = abs(float(spherical_jn(order, root)))
            print(f"  x_{order}{k} = {root:.12f}   |j_{order}(x)| = {residual:.2e}")
            entries.append({"index": k, "x": root, "residual": residual})
        payload[f"j{order}"] = entries

    with open(os.path.join(RESULTS, "zeros.json"), "w") as f:
        json.dump(payload, f, indent=2)

    lines = ["Zeros of the spherical Bessel functions", ""]
    for order, roots in sorted(zeros.items()):
        for k, root in enumerate(roots, start=1):
            lines.append(f"j{order}{k} = {root:.12f}")
    with open(os.path.join(RESULTS, "zeros.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    plot(zeros, os.path.join(RESULTS, "bessel_zeros.png"), x_max=args.x_max)
    print(f"\nWrote {RESULTS}")


if __name__ == "__main__":
    main()
