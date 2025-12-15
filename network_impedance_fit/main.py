"""Fit a small neural network to a transmission-line impedance curve,
training it with a derivative-free optimiser instead of backpropagation.

The target is the magnitude of the input impedance of a lossless line of
characteristic impedance Zc terminated in a load Zp:

    Z(x) = Zc * (Zp + j*Zc*tan(beta*x)) / (Zc + j*Zp*tan(beta*x))

The model is a 1-5-1 network with tanh hidden units -- 16 parameters, fitted by
Nelder-Mead simplex search on the mean squared error. Nelder-Mead never sees a
gradient, which is the point of the exercise: it treats the network as an
opaque function of 16 numbers. It is also why the fit needs random restarts,
since the simplex readily settles into a local minimum.

Weights are clipped to [-30, 30] inside the objective. That bounds the search
region, at the cost of making the landscape flat wherever the clip is active.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

HIDDEN = 5
N_PARAMS = 3 * HIDDEN + 1
WEIGHT_CLIP = 30.0

ZC = 5.0
ZP = 10.0 + 2j
BETA = 2.0 * np.pi

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def target(x):
    """Impedance magnitude along the line."""
    tan_bx = np.tan(BETA * x)
    numerator = ZP + ZC * tan_bx * 1j
    denominator = ZC + ZP * tan_bx * 1j
    return np.abs(ZC * numerator / denominator)


def forward(x, w):
    """1-5-1 network with tanh hidden units."""
    w1, b1, w2 = w[:HIDDEN], w[HIDDEN:2 * HIDDEN], w[2 * HIDDEN:3 * HIDDEN]
    return np.tanh(np.outer(x, w1) + b1) @ w2 + w[-1]


def mse(w, x, y):
    return float(np.mean((forward(x, np.clip(w, -WEIGHT_CLIP, WEIGHT_CLIP)) - y) ** 2))


def train(x, y, restarts=30, tolerance=0.0, seed=0):
    """Nelder-Mead from random starts, keeping the best.

    Restarts are not optional here. A single simplex lands anywhere between
    MSE 1e-2 and 1e-3 depending on where it starts, so the spread across
    restarts is part of the result rather than noise to hide. `tolerance`
    stops the sweep early once a fit is good enough; it defaults to 0, meaning
    run every restart and keep the best.
    """
    rng = np.random.default_rng(seed)
    best_w, best_mse = None, np.inf

    for attempt in range(restarts):
        w0 = rng.uniform(-1, 1, N_PARAMS)
        result = minimize(mse, w0, args=(x, y), method="nelder-mead",
                          options={"maxiter": 20000, "maxfev": 20000,
                                   "xatol": 1e-10, "fatol": 1e-12})
        w = np.clip(result.x, -WEIGHT_CLIP, WEIGHT_CLIP)
        score = mse(w, x, y)
        print(f"  restart {attempt + 1:2d}/{restarts}  MSE {score:.6e}")

        if score < best_mse:
            best_w, best_mse = w, score
        if tolerance > 0 and best_mse < tolerance:
            break

    return best_w, best_mse


def plot(x, y, w, filename):
    fig, (ax, ax_err) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1]})
    prediction = forward(x, w)
    ax.plot(x, y, linewidth=2, label="analytic $|Z(x)|$")
    ax.plot(x, prediction, "--", linewidth=2, label="network fit")
    ax.set_ylabel("impedance magnitude")
    ax.set_title("Transmission-line impedance, fitted by Nelder-Mead")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax_err.plot(x, prediction - y, linewidth=1.5)
    ax_err.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax_err.set_xlabel("distance along line")
    ax_err.set_ylabel("residual")
    ax_err.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--restarts", type=int, default=30)
    ap.add_argument("--tolerance", type=float, default=0.0,
                    help="stop early once MSE is below this (0 = run all restarts)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    x = np.arange(0, 0.25, 0.005)
    y = target(x)

    w, score = train(x, y, restarts=args.restarts,
                     tolerance=args.tolerance, seed=args.seed)
    print(f"\nBest MSE: {score:.15e}")
    print(f"Max absolute residual: {np.max(np.abs(forward(x, w) - y)):.6e}")

    with open(os.path.join(RESULTS, "weights.json"), "w") as f:
        json.dump({"weights": w.tolist(), "mse": score, "hidden_units": HIDDEN,
                   "restarts": args.restarts, "seed": args.seed,
                   "zc": ZC, "zp": [ZP.real, ZP.imag]}, f, indent=2)

    plot(x, y, w, os.path.join(RESULTS, "impedance_fit.png"))
    print(f"Wrote {RESULTS}")


if __name__ == "__main__":
    main()
