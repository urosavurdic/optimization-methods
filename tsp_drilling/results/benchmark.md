# Drilling path: exhaustive search vs 2-opt

Open path, L2 distances, coordinates in millimetres.

| holes | exhaustive | orderings | time | 2-opt | restarts | time | gap |
|---|---|---|---|---|---|---|---|
| 8 | 81.49 | 20,160 | 0.01s | 81.49 | 50 | 0.002s | -0.00% |
| 12 | 130.16 | 239,500,800 | 253.82s | 130.16 | 50 | 0.004s | 0.00% |
| 20 | not enumerable | ~1.2e18 | - | 205.13 | 50 | 0.011s | - |
