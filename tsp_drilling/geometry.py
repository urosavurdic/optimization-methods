"""Geometry helpers for the drilling-path problem.

Path length convention
----------------------
A path here is *open*: `path_length` sums the edges between consecutive holes
and does not add a closing edge back to the start. That is deliberate. The
drill does not fly home when it finishes a board -- it retraces its path
backwards -- so the quantity worth minimising is the one-way travel, not a
closed tour. This differs from the textbook TSP objective, and a tour that is
optimal open is not necessarily optimal closed.
"""

import json
import math
import random

NUM_POINTS = 20


def euclidean_distance(p, q):
    """L2 distance between two (x, y) points."""
    return math.hypot(p[0] - q[0], p[1] - q[1])


def path_length(path, points):
    """Total travel along `path`, an open sequence of indices into `points`."""
    return sum(
        euclidean_distance(points[path[i]], points[path[i + 1]])
        for i in range(len(path) - 1)
    )


def load_points(file_path, num_points=NUM_POINTS):
    """Load `num_points` hole coordinates from JSON."""
    with open(file_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    items = list(raw.items())
    sampled = random.sample(items, num_points)
    return [tuple(v) for _, v in sampled]


def describe(path, points, label="Path"):
    """Print a path and its length, and return the length."""
    length = path_length(path, points)
    print(f"{label}: {tuple(path)}")
    print(f"Length: {length:.2f}")
    return length
