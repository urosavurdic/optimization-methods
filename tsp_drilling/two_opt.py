"""2-opt local search for the open drilling path.

2-opt is the local-search core that Lin-Kernighan generalises: repeatedly pick
two positions, reverse the segment between them, and keep the change if the
path gets shorter. Lin-Kernighan's contribution is making the number of edges
exchanged variable rather than fixed at two; this implementation stays at the
fixed-2 case, which is enough to show the trade-off against exhaustive search.

On an *open* path the usual closed-tour shortcut does not apply. Reversing
`path[i:j+1]` rewires the edge entering the segment and the edge leaving it,
but when the segment touches either end of the path there is no such edge, so
the delta is computed from the affected edges directly rather than from a
fixed four-edge formula.
"""

import random
import time

from geometry import euclidean_distance, path_length


def _distance_matrix(points):
    n = len(points)
    return [[euclidean_distance(points[i], points[j]) for j in range(n)] for i in range(n)]


def _reversal_gain(path, dist, i, j):
    """Change in length from reversing path[i:j+1]. Negative means shorter."""
    before = 0.0
    after = 0.0
    if i > 0:
        before += dist[path[i - 1]][path[i]]
        after += dist[path[i - 1]][path[j]]
    if j < len(path) - 1:
        before += dist[path[j]][path[j + 1]]
        after += dist[path[i]][path[j + 1]]
    return after - before


def two_opt(points, path=None, rng=None):
    """Improve `path` until no single reversal helps. Returns (path, length)."""
    n = len(points)
    if n < 3:
        p = list(range(n)) if path is None else list(path)
        return p, path_length(p, points)

    dist = _distance_matrix(points)
    if path is None:
        path = list(range(n))
        (rng or random).shuffle(path)
    else:
        path = list(path)

    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                if _reversal_gain(path, dist, i, j) < -1e-12:
                    path[i:j + 1] = reversed(path[i:j + 1])
                    improved = True
    return path, path_length(path, points)


def two_opt_multistart(points, restarts=50, seed=0):
    """Run 2-opt from many random starts and keep the best result."""
    rng = random.Random(seed)
    best_path, best_length = None, float("inf")
    start = time.perf_counter()

    for _ in range(restarts):
        path, length = two_opt(points, rng=rng)
        if length < best_length:
            best_path, best_length = path, length

    return {
        "path": tuple(best_path),
        "length": best_length,
        "restarts": restarts,
        "time": time.perf_counter() - start,
    }


if __name__ == "__main__":
    import os
    from geometry import load_points

    here = os.path.dirname(os.path.abspath(__file__))
    for n in (8, 12, 20):
        points = load_points(os.path.join(here, "coordinates.json"), num_points=n)
        result = two_opt_multistart(points, restarts=50)
        print(f"n={n:2d}  length {result['length']:7.2f}  "
              f"{result['restarts']} restarts in {result['time']:.3f}s  {result['path']}")
