"""Genetic algorithm on the memory-packing problem.

Binary tournament selection, single-point crossover, single-bit mutation.

Elitism is off by default, which is the configuration the original experiment
settled on: tournament selection alone gave more stable convergence across
repeated runs than elitism did. Turning elitism off means the best individual
can be bred out of the population between generations, so the champion is
tracked *outside* the population -- `best_x` and `best_score` are updated
together, and an assertion at the end enforces that the reported score really
is the score of the reported solution. Tracking a running-minimum score while
returning the final generation's best is how the two silently drift apart.
"""

import argparse
import json
import os

import numpy as np

from problem import DIM, BUDGET, OBJECTIVES, report

POP_SIZE = 2000
NUM_GEN = 50
CROSS_PROB = 0.8
MUT_PROB = 0.1
ELITE_FRAC = 0.2
RUNS = 20


def _tournament(pop, fitness, rng):
    i, j = rng.choice(len(pop), 2, replace=False)
    return pop[i] if fitness[i] <= fitness[j] else pop[j]


def evolve(objective, rng, pop_size=POP_SIZE, generations=NUM_GEN,
           cross_prob=CROSS_PROB, mut_prob=MUT_PROB, elitism=False,
           elite_frac=ELITE_FRAC):
    """One GA run. Returns (best_x, best_score, history)."""
    f = OBJECTIVES[objective]

    pop = rng.integers(0, 2, (pop_size, DIM))
    fitness = np.array([f(ind) for ind in pop])

    champion_idx = int(np.argmin(fitness))
    best_x, best_score = pop[champion_idx].copy(), int(fitness[champion_idx])

    history = []
    elite_size = int(pop_size * elite_frac) if elitism else 0

    for _ in range(generations):
        new_pop = []
        if elite_size:
            new_pop.extend(pop[np.argsort(fitness)[:elite_size]])

        while len(new_pop) < pop_size:
            p1 = _tournament(pop, fitness, rng)
            p2 = _tournament(pop, fitness, rng)

            if rng.random() < cross_prob:
                cut = rng.integers(1, DIM)
                c1 = np.concatenate((p1[:cut], p2[cut:]))
                c2 = np.concatenate((p2[:cut], p1[cut:]))
            else:
                c1, c2 = p1.copy(), p2.copy()

            for child in (c1, c2):
                if rng.random() < mut_prob:
                    child[rng.integers(0, DIM)] ^= 1
                if len(new_pop) < pop_size:
                    new_pop.append(child)

        pop = np.array(new_pop)
        fitness = np.array([f(ind) for ind in pop])

        # Champion is kept outside the population, so it survives even when a
        # generation breeds it away.
        gen_best = int(np.argmin(fitness))
        if fitness[gen_best] < best_score:
            best_x, best_score = pop[gen_best].copy(), int(fitness[gen_best])
        history.append(best_score)

    return best_x, best_score, np.array(history)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--objective", choices=sorted(OBJECTIVES), default="slack")
    ap.add_argument("--runs", type=int, default=RUNS)
    ap.add_argument("--generations", type=int, default=NUM_GEN)
    ap.add_argument("--population", type=int, default=POP_SIZE)
    ap.add_argument("--elitism", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(here, "results")
    os.makedirs(results_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    histories, best_x, best_score = [], None, float("inf")

    for run in range(args.runs):
        x, score, history = evolve(args.objective, rng, pop_size=args.population,
                                   generations=args.generations, elitism=args.elitism)
        histories.append(history)
        if score < best_score:
            best_x, best_score = x.copy(), score
        print(f"run {run + 1:2d}/{args.runs}  best {score}")

    summary = report(best_x, args.objective)
    assert summary["score"] == best_score, "best solution and best score disagree"

    print(f"\nBest {args.objective}: {best_score}")
    print(f"Selected {summary['selected']}/{DIM} requests, "
          f"{summary['total_bytes']:,} of {BUDGET:,} bytes, fits={summary['fits']}")

    summary.update(runs=args.runs, generations=args.generations,
                   population=args.population, elitism=args.elitism, seed=args.seed)
    tag = args.objective + ("_elitism" if args.elitism else "")
    with open(os.path.join(results_dir, f"genetic_{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    np.save(os.path.join(results_dir, f"genetic_{tag}_history.npy"),
            np.minimum.accumulate(np.array(histories), axis=1))


if __name__ == "__main__":
    main()
