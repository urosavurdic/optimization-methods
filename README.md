# Optimization methods

Six optimization problems, each solved from scratch in NumPy — no autograd, no
solver library, no metaheuristic package. Where a problem is small enough to
settle exactly, the exact answer is computed too, so the heuristics are scored
against ground truth rather than against each other.

This began as university coursework. It is kept because the implementations are
the point.

```bash
python -m venv .venv && .venv/Scripts/activate     # Windows
pip install -r requirements.txt
```

Every module writes figures and machine-readable results into its own
`results/` directory, and every module runs standalone.

---

## Drilling path — exhaustive search against 2-opt

`tsp_drilling/` · [results](tsp_drilling/results/benchmark.md)

A drill visits every hole on a circuit board once. Minimising its travel is a
travelling-salesman problem with one twist: the drill does not return to the
first hole, it retraces its path, so the objective is an **open** path — the
sum of `n-1` edges, with no closing edge. A tour that is optimal closed is not
necessarily optimal open, so the standard formulation would answer a different
question.

| holes | exhaustive | orderings | time | 2-opt | time | gap |
|---|---|---|---|---|---|---|
| 8 | **81.49** | 20,160 | 0.01 s | **81.49** | 0.002 s | 0% |
| 12 | **130.16** | 239,500,800 | 254 s | **130.16** | 0.004 s | 0% |
| 20 | not enumerable | ~1.2×10¹⁸ | — | 205.13 | 0.011 s | — |

2-opt finds the exact optimum on both instances where the optimum is knowable,
about 70,000× faster at 12 holes. That is the entire argument for heuristics on
this problem class, and it is worth noting what it does *not* prove: matching
the optimum at n=12 is evidence, not a guarantee, and at n=20 there is no way
to check.

Exhaustive search only scores orderings that start at the lower-indexed
endpoint, since a path and its reverse have equal length, and abandons a
partial ordering as soon as it exceeds the best complete one. Both together cut
the 12-hole search from 479,001,600 orderings to 239,500,800 and from roughly
90 minutes to 4.

2-opt is the local-search core that Lin–Kernighan generalises: reverse a
segment, keep the change if the path shortens. Lin–Kernighan's contribution is
making the number of exchanged edges variable; this stays at the fixed-2 case.

## Memory packing — simulated annealing against a genetic algorithm

`memory_packing/` · [results](memory_packing/results/)

Sixty-four memory requests, one 2²⁶-byte budget, choose the subset that fills
it best. Subset-sum: NP-hard in general, small enough here to get close but not
to prove optimality.

**The objective function matters more than the search method.** Two are
implemented and their scores are not comparable:

- `slack` — budget minus total, rejecting any overshoot. Every feasible
  solution actually fits in memory.
- `deviation` — absolute distance to the budget, overshoot permitted. Scores
  lower, but a winning solution need not fit.

| method | `slack` | `deviation` |
|---|---|---|
| simulated annealing | 7 bytes | 1 byte |
| genetic algorithm, tournament | **4 bytes** | **0 bytes** |
| genetic algorithm, elitism | 18 bytes | — |

The GA lands on the budget exactly under `deviation` — 67,108,864 of
67,108,864 bytes — and comes within 4 bytes under the stricter `slack`
objective. `deviation` scores better mostly because it is an easier landscape,
not because the search is better.

Elitism scores 18 against tournament selection's 4. That reproduces the
original finding: preserving the top fraction each generation converges faster
but lands worse, because the population loses the diversity it needs to escape
the local optimum it found first.

Both solvers assert, before writing results, that the score they report is the
score of the solution they return. Without that check a search that tracks a
running-minimum score while returning the final generation's best will report a
number no committed solution achieves.

## Backpropagation from scratch, raced against a genetic algorithm

`gd_vs_ga/` · [results](gd_vs_ga/results/)

A 784-32-10 network on MNIST, forward and backward passes written by hand.
Starting from identical weights, the same network is trained by gradient
descent and by a genetic algorithm that never computes a gradient.

| | test accuracy | test loss | wall clock |
|---|---|---|---|
| gradient descent, 10 epochs | **93.98%** | 0.209 | **1.1 s** |
| genetic algorithm, 10 generations | 74.37% | 0.799 | 56.4 s |

Gradient descent passes the GA's *final* accuracy inside its first epoch, after
0.1 seconds — roughly 500× less compute for a better answer.

Both are plotted against **wall-clock time rather than epochs**. A generation
costs 25 forward passes per batch where an epoch costs one forward and one
backward pass, so an epoch-indexed plot would flatter the GA for free.

The gap is not a tuning artefact. With 25,450 parameters, the gradient carries
25,450 numbers of information per batch; the GA's loss value carries one. This
is a demonstration of why backpropagation matters, not a contest with an
uncertain outcome.

MNIST is read straight from the IDX files by `mnist.py` — using a deep-learning
framework as a file reader would undercut the premise.

## Zeros of the spherical Bessel functions

`bessel_zeros/` · [results](bessel_zeros/results/zeros.txt)

Bisection is the slowest root-finder in common use — one bit per iteration —
and the only one that cannot diverge given a sign change. That makes it the
right tool when the task is finding *every* root in a range rather than
polishing one quickly. A dense grid brackets each sign change, then bisection
runs to full double precision.

The first two roots of each order, to 12 decimal places:

| | first | second |
|---|---|---|
| j₁ | 4.493409457909 | 7.725251836937 |
| j₂ | 5.763459196894 | 9.095011330476 |

Residuals `|jₙ(x)|` at the returned roots stay below 6×10⁻¹⁴, and the
values agree with `scipy.optimize.brentq` to within 7×10⁻¹³.

## Transmission-line impedance, fitted without gradients

`network_impedance_fit/` · [results](network_impedance_fit/results/)

The magnitude of the input impedance of a lossless line terminated in a complex
load, fitted by a 1-5-1 tanh network whose 16 parameters are found by
Nelder-Mead simplex search. Nelder-Mead never sees a gradient — it treats the
network as an opaque function of 16 numbers.

Best of 30 random restarts: **MSE 8.80×10⁻⁵**, maximum absolute residual
3.08×10⁻².

Restarts are not optional. A single simplex lands anywhere between 10⁻² and
10⁻⁴ depending on where it starts, so the spread across restarts is part of the
result rather than noise to suppress.

## Source localization by differential evolution

`source_localization/` · [results](source_localization/results/)

Twenty sensors on a circle read the superposition of two 1/r sources somewhere
inside. Recover both positions and both amplitudes — six unknowns — from the
twenty readings.

The inverse problem is badly conditioned: swapping the two sources, or trading
amplitude against distance, produces nearly identical readings, so the
least-squares surface is full of local minima. Differential evolution suits it
because its mutation step — perturb one member by the scaled difference of two
others — adapts its own step size to the spread of the population, exploring
widely early and refining late with no schedule to tune.

Best of 10 runs: **residual sum of squares 2.57×10⁻³**, with sources at
(−9.05, 4.89) amplitude −0.92 and (7.88, −7.63) amplitude 2.84. Because the
labelling of the two sources is arbitrary, a solution and its swap are the same
answer.

---

Uroš Savurdić — School of Electrical Engineering, University of Belgrade.
