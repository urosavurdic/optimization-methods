"""Minimal MNIST loader: download, cache, parse IDX, return NumPy arrays.

The point of this project is that nothing here leans on a deep-learning
framework, so pulling in torchvision purely to read four binary files would
undercut it. The IDX format is a 16-byte header followed by raw bytes, which
is a few lines to parse directly.
"""

import gzip
import os
import struct
import urllib.request

import numpy as np

# The canonical LeCun URLs now refuse unauthenticated requests; this is the
# same mirror torchvision falls back to.
BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"

FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _download(name, root):
    """Return the path to `name`, fetching it only if it is not already cached."""
    os.makedirs(root, exist_ok=True)
    gz_path = os.path.join(root, name)
    raw_path = gz_path[:-3]

    if os.path.exists(raw_path) or os.path.exists(gz_path):
        return gz_path if os.path.exists(gz_path) else raw_path

    print(f"  downloading {name} ...")
    urllib.request.urlretrieve(BASE_URL + name, gz_path)
    return gz_path


def _read_idx(path):
    """Parse an IDX file, gzipped or not, into a NumPy array."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        dims = magic & 0xFF          # low byte of the magic number is the rank
        shape = (count,) + tuple(
            struct.unpack(">I", f.read(4))[0] for _ in range(dims - 1)
        )
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


def load_mnist(root):
    """Return (train_X, train_y, test_X, test_y).

    Images arrive as float32 in [0, 1], flattened to 784 features per sample,
    matching what `transforms.ToTensor()` would have produced.
    """
    out = {}
    for key, name in FILES.items():
        array = _read_idx(_download(name, root))
        if key.endswith("images"):
            out[key] = (array.reshape(len(array), -1).astype(np.float32) / 255.0)
        else:
            out[key] = array.astype(np.int64)

    return out["train_images"], out["train_labels"], out["test_images"], out["test_labels"]


def iterate_batches(X, y, batch_size, rng=None):
    """Split X, y into batches, shuffled when `rng` is given."""
    order = np.arange(len(X))
    if rng is not None:
        rng.shuffle(order)
    return [(X[order[i:i + batch_size]], y[order[i:i + batch_size]])
            for i in range(0, len(order), batch_size)]
