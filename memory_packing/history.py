"""Saving and loading convergence histories.

A 100,000-iteration annealing run over 20 repeats is 16 MB of float64 if stored
raw, which is more than the rest of this repository put together and far more
resolution than any plot can show. The curves are drawn on a log x-axis, so the
samples that matter are dense early and sparse late -- exactly what log-spaced
indices give. Storing ~1,200 of those per run keeps every visible feature and
costs a few hundred kilobytes.

Files are `.npz`: `steps` holds the iteration index of each sample, `runs` holds
one row of best-so-far values per run.
"""

import os

import numpy as np

MAX_SAMPLES = 1200


def log_spaced_indices(length, max_samples=MAX_SAMPLES):
    """Indices from 0 to length-1, dense at the start, thinning geometrically."""
    if length <= max_samples:
        return np.arange(length)
    idx = np.unique(np.geomspace(1, length, num=max_samples).astype(np.int64) - 1)
    return np.clip(idx, 0, length - 1)


def save_histories(path, histories):
    """Store the running minimum of each run, sampled on a log grid."""
    runs = np.minimum.accumulate(np.asarray(histories, dtype=np.float64), axis=1)
    steps = log_spaced_indices(runs.shape[1])
    np.savez_compressed(path, steps=steps, runs=runs[:, steps].astype(np.float32))


def load_histories(path):
    """Return (steps, runs). Accepts the legacy raw .npy layout too."""
    if path.endswith(".npy") or not os.path.exists(path):
        legacy = path.replace(".npz", ".npy")
        if os.path.exists(legacy):
            runs = np.load(legacy)
            return np.arange(runs.shape[1]), runs
    data = np.load(path)
    return data["steps"], data["runs"]
