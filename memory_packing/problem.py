"""The shared subset-selection problem solved by both search methods here.

Sixty-four memory requests, one 2^26-byte budget. Choose a subset whose total
is as close to the budget as possible without a solver -- a subset-sum problem,
NP-hard in general, small enough here that a good heuristic answer is findable
but a proof of optimality is not.

Two objectives appear in the literature and both are implemented, because they
are *not* interchangeable and the scores they produce cannot be compared:

- `slack`     -- budget minus the total, with overshoot rejected outright.
                 Every feasible solution actually fits in memory.
- `deviation` -- absolute distance to the budget, overshoot allowed.
                 Scores lower, but a winning solution may not fit.

`slack` is the one that answers the engineering question. `deviation` is the
easier search landscape, which is why it finds smaller numbers.
"""

import numpy as np

SIZES = np.array([
    173669, 275487, 1197613, 1549805, 502334, 217684, 1796841, 274708,
    631252, 148665, 150254, 4784408, 344759, 440109, 4198037, 329673,
    28602, 144173, 1461469, 187895, 369313, 959307, 1482335, 2772513,
    1313997, 254845, 486167, 2667146, 264004, 297223, 94694, 1757457,
    576203, 8577828, 498382, 8478177, 123575, 4062389, 3001419, 196884,
    617991, 421056, 3017627, 131936, 1152730, 2676649, 656678, 4519834,
    201919, 56080, 2142553, 326263, 8172117, 2304253, 4761871, 205387,
    6148422, 414559, 2893305, 2158562, 465972, 304078, 1841018, 1915571
])

DIM = len(SIZES)
BUDGET = 2 ** 26


def slack(x):
    """Budget minus selected total; BUDGET (worst score) if over budget."""
    remaining = BUDGET - int(np.dot(SIZES, x))
    return remaining if remaining >= 0 else BUDGET


def deviation(x):
    """Absolute distance to the budget. Permits exceeding it."""
    return abs(BUDGET - int(np.dot(SIZES, x)))


OBJECTIVES = {"slack": slack, "deviation": deviation}


def report(x, objective="slack"):
    """Human-readable summary of a candidate solution."""
    total = int(np.dot(SIZES, x))
    return {
        "bits": "".join(map(str, np.asarray(x, dtype=int))),
        "selected": int(np.sum(x)),
        "total_bytes": total,
        "budget_bytes": BUDGET,
        "slack_bytes": BUDGET - total,
        "fits": total <= BUDGET,
        "score": OBJECTIVES[objective](x),
        "objective": objective,
    }
