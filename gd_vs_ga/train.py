"""Backpropagation from scratch, raced against a genetic algorithm.

A 784-32-10 network on MNIST, forward and backward passes written by hand in
NumPy -- no autograd. The same network, from the same initial weights, is then
trained two ways:

  gradient descent   -- uses the analytic gradient, one update per mini-batch
  genetic algorithm  -- never computes a gradient; it evaluates the loss of a
                        population and recombines whatever scores well

Both curves are plotted against **wall-clock time, not epochs**. An epoch and a
generation cost wildly different amounts of compute -- a GA generation here is
POP_SIZE forward passes per batch against gradient descent's single forward and
backward pass -- so an epoch-indexed plot would flatter the GA for free. Time
is the axis on which the comparison is fair.

The expected result is that gradient descent wins by a wide margin, and the
reason is worth stating plainly: with 25,450 parameters, the gradient supplies
25,450 numbers of information per batch, while the GA's loss value supplies
one. This is a demonstration of *why* backpropagation matters, not a contest
with an uncertain outcome.
"""

import argparse
import csv
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mnist import iterate_batches, load_mnist

D_IN, D_HIDDEN, D_OUT = 784, 32, 10
PARAM_SIZE = D_IN * D_HIDDEN + D_HIDDEN + D_HIDDEN * D_OUT + D_OUT
BATCH = 256

POP_SIZE = 25
CROSS_PROB = 0.8
MUT_PROB = 0.2
MUTATION_SCALE = 0.05
MUTATION_FRACTION = 10  # mutate one tenth of the genome

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
DATA = os.path.join(HERE, "data")


def unpack(w):
    """Split the flat parameter vector into weight matrices and bias vectors."""
    i = 0
    W1 = w[i:i + D_IN * D_HIDDEN].reshape(D_IN, D_HIDDEN); i += D_IN * D_HIDDEN
    b1 = w[i:i + D_HIDDEN]; i += D_HIDDEN
    W2 = w[i:i + D_HIDDEN * D_OUT].reshape(D_HIDDEN, D_OUT); i += D_HIDDEN * D_OUT
    b2 = w[i:i + D_OUT]
    return W1, b1, W2, b2


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)   # shift for numerical stability
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def forward_loss(w, X, y):
    """Forward pass returning cross-entropy loss and cached activations."""
    W1, b1, W2, b2 = unpack(w)
    z1 = X @ W1 + b1
    a1 = np.maximum(0, z1)
    z2 = a1 @ W2 + b2
    p = softmax(z2)
    loss = -np.mean(np.log(p[np.arange(len(y)), y] + 1e-9))
    return loss, (z1, a1, z2, p)


def backward(w, X, y, cache):
    """Analytic gradient of the loss w.r.t. the flat parameter vector."""
    _, _, W2, _ = unpack(w)
    z1, a1, _, p = cache

    dz2 = p.copy()
    dz2[np.arange(len(y)), y] -= 1
    dz2 /= len(y)

    dW2 = a1.T @ dz2
    db2 = dz2.sum(0)
    dz1 = (dz2 @ W2.T) * (z1 > 0)
    dW1 = X.T @ dz1
    db1 = dz1.sum(0)

    return np.concatenate([dW1.ravel(), db1, dW2.ravel(), db2])


def evaluate(w, batches):
    """Mean loss and accuracy over a list of (X, y) batches."""
    total_loss, correct, total = 0.0, 0, 0
    for X, y in batches:
        loss, cache = forward_loss(w, X, y)
        total_loss += loss * len(y)
        correct += int((np.argmax(cache[2], axis=1) == y).sum())
        total += len(y)
    return total_loss / total, correct / total


def count_parameters():
    return PARAM_SIZE


def train_gradient_descent(w_init, train_batches, test_batches, epochs, lr):
    w = w_init.copy()
    loss, acc = evaluate(w, test_batches)
    times, losses, accuracies = [0.0], [loss], [acc]
    print(f"GD  epoch  0 | loss {loss:.4f} | acc {acc:.4f}")

    start = time.perf_counter()
    for epoch in range(epochs):
        for X, y in train_batches:
            _, cache = forward_loss(w, X, y)
            w -= lr * backward(w, X, y, cache)

        loss, acc = evaluate(w, test_batches)
        times.append(time.perf_counter() - start)
        losses.append(loss)
        accuracies.append(acc)
        print(f"GD  epoch {epoch + 1:2d} | loss {loss:.4f} | acc {acc:.4f} "
              f"| {times[-1]:.1f}s")

    return times, losses, accuracies


def _tournament(pop, fitness, rng):
    i, j = rng.choice(len(pop), 2, replace=False)
    return pop[i] if fitness[i] <= fitness[j] else pop[j]


def train_genetic(w_init, train_batches, test_batches, generations, rng, py_rng):
    """Arithmetic-crossover GA over the same parameter vector."""
    pop = [w_init.copy()] + [rng.standard_normal(PARAM_SIZE).astype(np.float32) * 0.01
                             for _ in range(POP_SIZE - 1)]

    loss, acc = evaluate(pop[0], test_batches)
    times, losses, accuracies = [0.0], [loss], [acc]
    print(f"GA  gen    0 | loss {loss:.4f} | acc {acc:.4f}")

    start = time.perf_counter()
    for generation in range(generations):
        for batch in train_batches:
            fitness = np.array([forward_loss(ind, *batch)[0] for ind in pop])
            new_pop = []

            while len(new_pop) < POP_SIZE:
                p1 = _tournament(pop, fitness, rng)
                p2 = _tournament(pop, fitness, rng)

                # A converged population can make every draw identical, so cap
                # the retries instead of spinning forever.
                for _ in range(10):
                    if not np.array_equal(p1, p2):
                        break
                    p2 = _tournament(pop, fitness, rng)

                if py_rng.random() < CROSS_PROB:
                    alpha = py_rng.uniform(0, 1)
                    children = [p2 + alpha * (p1 - p2),
                                p2 + alpha * (p2 - p1),
                                p1 + alpha * (p1 - p2)]
                else:
                    children = [p1.copy(), p2.copy()]

                for child in children:
                    if py_rng.random() < MUT_PROB:
                        idx = rng.choice(PARAM_SIZE, PARAM_SIZE // MUTATION_FRACTION,
                                         replace=False)
                        child[idx] += rng.standard_normal(len(idx)) * MUTATION_SCALE

                order = np.argsort([forward_loss(c, *batch)[0] for c in children])
                for k in order:
                    if len(new_pop) < POP_SIZE:
                        new_pop.append(children[k])

            pop = new_pop

        scores = [evaluate(ind, test_batches)[0] for ind in pop]
        best = pop[int(np.argmin(scores))]
        loss, acc = evaluate(best, test_batches)
        times.append(time.perf_counter() - start)
        losses.append(loss)
        accuracies.append(acc)
        print(f"GA  gen {generation + 1:3d} | loss {loss:.4f} | acc {acc:.4f} "
              f"| {times[-1]:.1f}s")

    return times, losses, accuracies


def plot(gd, ga, filename):
    (gd_t, gd_loss, gd_acc), (ga_t, ga_loss, ga_acc) = gd, ga
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(gd_t, gd_loss, "-o", linewidth=2, markersize=4, label="gradient descent")
    axes[0].plot(ga_t, ga_loss, "-s", linewidth=2, markersize=4, label="genetic algorithm")
    axes[0].set_ylabel("test cross-entropy loss")
    axes[0].set_title("Loss against wall-clock time")

    axes[1].plot(gd_t, gd_acc, "-o", linewidth=2, markersize=4, label="gradient descent")
    axes[1].plot(ga_t, ga_acc, "-s", linewidth=2, markersize=4, label="genetic algorithm")
    axes[1].set_ylabel("test accuracy")
    axes[1].set_title("Accuracy against wall-clock time")

    for ax in axes:
        ax.set_xlabel("seconds")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("784-32-10 network on MNIST, identical initial weights")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--generations", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    py_rng = random.Random(args.seed)

    train_X, train_y, test_X, test_y = load_mnist(DATA)
    train_batches = iterate_batches(train_X, train_y, BATCH, rng=rng)
    test_batches = iterate_batches(test_X, test_y, BATCH)
    print(f"{len(train_batches)} training batches, {len(test_batches)} test batches, "
          f"{PARAM_SIZE:,} parameters\n")

    w_init = rng.standard_normal(PARAM_SIZE).astype(np.float32) * 0.01

    gd = train_gradient_descent(w_init, train_batches, test_batches, args.epochs, args.lr)
    print()
    ga = train_genetic(w_init, train_batches, test_batches, args.generations, rng, py_rng)

    plot(gd, ga, os.path.join(RESULTS, "curves.png"))

    with open(os.path.join(RESULTS, "history.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "step", "seconds", "test_loss", "test_accuracy"])
        for name, (times, losses, accuracies) in (("gradient_descent", gd),
                                                  ("genetic_algorithm", ga)):
            for step, (t, l, a) in enumerate(zip(times, losses, accuracies)):
                writer.writerow([name, step, f"{t:.4f}", f"{l:.6f}", f"{a:.6f}"])

    print(f"\nGD final: loss {gd[1][-1]:.4f} | acc {gd[2][-1]:.4f} | {gd[0][-1]:.1f}s")
    print(f"GA final: loss {ga[1][-1]:.4f} | acc {ga[2][-1]:.4f} | {ga[0][-1]:.1f}s")
    print(f"Wrote {RESULTS}")


if __name__ == "__main__":
    main()
