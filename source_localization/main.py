"""Locate two sources from boundary measurements, using differential evolution.

Twenty sensors sit on a circle of radius R0. Each reads the superposition of
two 1/r sources placed somewhere inside:

    s_i = A1 / d(p_i, c1) + A2 / d(p_i, c2)

Given the twenty readings, recover both positions and both amplitudes -- six
unknowns. This is an inverse problem: the forward map is cheap, the inverse is
not, and the least-squares surface is riddled with local minima because
swapping the two sources, or trading amplitude against distance, produces
nearly identical readings.

Differential evolution suits it well. It needs no gradient, and its mutation
step -- perturb one member by the scaled difference of two others -- adapts its
own step size to the spread of the population, so it explores widely early and
refines late without a schedule.

Sources outside the sensor circle are rejected by returning a constant penalty,
which makes the infeasible region flat rather than merely expensive.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MEASUREMENTS = np.array([
    2.424595205726587e-01, 1.737226395065819e-01, 1.315612759386036e-01,
    1.022985539042393e-01, 7.905975891960761e-02, 5.717509542148174e-02,
    3.155886625106896e-02, -6.242228581847679e-03, -6.565183775481365e-02,
    -8.482380513926287e-02, -1.828677714588237e-02, 3.632382803076845e-02,
    7.654845872485493e-02, 1.152250132891757e-01, 1.631742367154961e-01,
    2.358469152696193e-01, 3.650430801728451e-01, 5.816044173713664e-01,
    5.827732223753571e-01, 3.686942505423780e-01
])

N_SENSORS = 20
R0 = 15.0
DIM = 6
PENALTY = 100.0

POP_SIZE = 60
GENERATIONS = 100
F_WEIGHT = 0.05
CROSS_PROB = 0.05
BOUND = 10.0

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

SENSORS = np.array([[R0 * np.cos(2 * np.pi * i / N_SENSORS),
                     R0 * np.sin(2 * np.pi * i / N_SENSORS)]
                    for i in range(N_SENSORS)])


def residual_sum_squares(v):
    """Least-squares misfit, or PENALTY if a source escapes the sensor circle."""
    if np.hypot(v[0], v[1]) >= R0 or np.hypot(v[2], v[3]) >= R0:
        return PENALTY

    d1 = np.hypot(SENSORS[:, 0] - v[0], SENSORS[:, 1] - v[1])
    d2 = np.hypot(SENSORS[:, 0] - v[2], SENSORS[:, 1] - v[3])
    predicted = v[4] / d1 + v[5] / d2
    return float(np.sum((predicted - MEASUREMENTS) ** 2))


def differential_evolution(rng, pop_size=POP_SIZE, generations=GENERATIONS,
                           f_weight=F_WEIGHT, cross_prob=CROSS_PROB, bound=BOUND):
    """Classic DE/rand/1/bin. Returns (best_vector, best_score, history)."""
    pop = rng.uniform(-bound, bound, (pop_size, DIM))
    scores = np.array([residual_sum_squares(v) for v in pop])
    history = []

    for _ in range(generations):
        for i in range(pop_size):
            choices = [j for j in range(pop_size) if j != i]
            a, b, c = rng.choice(choices, 3, replace=False)
            mutant = pop[a] + f_weight * (pop[b] - pop[c])

            # Binomial crossover: each coordinate takes the mutant with
            # probability cross_prob, and one forced coordinate guarantees the
            # trial differs from its parent.
            mask = rng.random(DIM) < cross_prob
            mask[rng.integers(DIM)] = True
            trial = np.where(mask, mutant, pop[i])

            trial_score = residual_sum_squares(trial)
            if trial_score < scores[i]:
                pop[i], scores[i] = trial, trial_score

        history.append(float(np.min(scores)))

    best = int(np.argmin(scores))
    return pop[best].copy(), float(scores[best]), np.array(history)


def plot(v, history, filename):
    fig, (ax_geom, ax_fit) = plt.subplots(1, 2, figsize=(13, 5.5))

    angle = np.linspace(0, 2 * np.pi, 400)
    ax_geom.plot(R0 * np.cos(angle), R0 * np.sin(angle), "--", alpha=0.4,
                 label=f"sensor circle (R={R0:.0f})")
    ax_geom.plot(SENSORS[:, 0], SENSORS[:, 1], "o", markersize=5, label="sensors")
    ax_geom.plot(v[0], v[1], "*", markersize=20, label=f"source 1 (A={v[4]:.2f})")
    ax_geom.plot(v[2], v[3], "*", markersize=20, label=f"source 2 (A={v[5]:.2f})")
    ax_geom.set_aspect("equal")
    ax_geom.set_title("Recovered source geometry")
    ax_geom.grid(True, alpha=0.3)
    ax_geom.legend(fontsize=8, loc="upper right")

    d1 = np.hypot(SENSORS[:, 0] - v[0], SENSORS[:, 1] - v[1])
    d2 = np.hypot(SENSORS[:, 0] - v[2], SENSORS[:, 1] - v[3])
    ax_fit.plot(MEASUREMENTS, "o-", label="measured")
    ax_fit.plot(v[4] / d1 + v[5] / d2, "s--", label="reconstructed")
    ax_fit.set_xlabel("sensor index")
    ax_fit.set_ylabel("reading")
    ax_fit.set_title("Measured vs reconstructed readings")
    ax_fit.grid(True, alpha=0.3)
    ax_fit.legend()

    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(history, linewidth=2)
    ax.set_xlabel("generation")
    ax.set_ylabel("sum of squared residuals")
    ax.set_title("Differential evolution convergence")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename.replace("geometry", "convergence"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--generations", type=int, default=GENERATIONS)
    ap.add_argument("--population", type=int, default=POP_SIZE)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    best_v, best_score, best_history = None, np.inf, None
    for run in range(args.runs):
        v, score, history = differential_evolution(
            rng, pop_size=args.population, generations=args.generations)
        print(f"run {run + 1:2d}/{args.runs}  residual {score:.6e}")
        if score < best_score:
            best_v, best_score, best_history = v, score, history

    print(f"\nBest residual sum of squares: {best_score:.16e}")
    print(f"  source 1: ({best_v[0]:+.4f}, {best_v[1]:+.4f})  amplitude {best_v[4]:+.4f}")
    print(f"  source 2: ({best_v[2]:+.4f}, {best_v[3]:+.4f})  amplitude {best_v[5]:+.4f}")
    assert np.isclose(residual_sum_squares(best_v), best_score), \
        "best solution and best score disagree"

    with open(os.path.join(RESULTS, "solution.json"), "w") as f:
        json.dump({
            "residual_sum_squares": best_score,
            "source_1": {"x": best_v[0], "y": best_v[1], "amplitude": best_v[4]},
            "source_2": {"x": best_v[2], "y": best_v[3], "amplitude": best_v[5]},
            "vector": best_v.tolist(),
            "runs": args.runs, "generations": args.generations,
            "population": args.population, "seed": args.seed,
        }, f, indent=2)

    plot(best_v, best_history, os.path.join(RESULTS, "source_geometry.png"))
    print(f"Wrote {RESULTS}")


if __name__ == "__main__":
    main()
