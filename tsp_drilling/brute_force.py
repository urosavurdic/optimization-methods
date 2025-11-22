"""Exhaustive search over all orderings of the holes.

Enumerates every permutation and keeps the shortest. This is the ground truth
the heuristic in `two_opt.py` is measured against, and it is only tractable for
small instances: 8 holes is 40,320 orderings, 12 holes is 479,001,600.
"""

import itertools
import time

from geometry import euclidean_distance, path_length


def _distance_matrix(points):
    n = len(points)
    return [[euclidean_distance(points[i], points[j]) for j in range(n)] for i in range(n)]


def brute_force_tsp(points, progress_every=5_000_000):
    """Return the shortest open path visiting every hole exactly once.

    Each ordering and its reverse have identical length, so only permutations
    starting at the lower-indexed endpoint are scored. That halves the work
    without changing the answer.
    """
    n = len(points)
    if n < 2:
        return {"path": tuple(range(n)), "length": 0.0, "permutations_checked": 1, "time": 0.0}

    dist = _distance_matrix(points)
    best_path, best_length, checked = None, float("inf"), 0
    start = time.perf_counter()

    for perm in itertools.permutations(range(n)):
        if perm[0] > perm[-1]:
            continue
        checked += 1

        length = 0.0
        for i in range(n - 1):
            length += dist[perm[i]][perm[i + 1]]
            if length >= best_length:
                break
        else:
            best_length, best_path = length, perm

        if progress_every and checked % progress_every == 0:
            print(f"  {checked:,} orderings, best {best_length:.2f}, "
                  f"{time.perf_counter() - start:.1f}s")

    return {
        "path": best_path,
        "length": best_length,
        "permutations_checked": checked,
        "time": time.perf_counter() - start,
    }


if __name__ == "__main__":
    import os
    from geometry import load_points

    here = os.path.dirname(os.path.abspath(__file__))
    points = load_points(os.path.join(here, "coordinates.json"), num_points=8)
    result = brute_force_tsp(points)
    print(f"Best path: {result['path']}")
    print(f"Length: {result['length']:.2f}")
    print(f"Orderings checked: {result['permutations_checked']:,}")
    print(f"Time: {result['time']:.2f}s")
