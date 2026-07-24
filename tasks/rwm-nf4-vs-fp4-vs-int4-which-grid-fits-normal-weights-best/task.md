## Context

The weights of a neural network are often well approximated by a standard normal distribution $w \sim \mathcal{N}(0,1)$. Quantising these real‑valued tensors to 4 bits is common in low‑precision inference. Three popular 4‑bit grids are:

- **NF4** – a *normalised* grid that places 16 evenly spaced points between $-1$ and $+1$.
- **FP4** – a *floating‑point* grid that places 16 evenly spaced points between $-8$ and $+8$.
- **INT4** – a signed integer grid with the values $\{-8,\dots,+7\}$.

For a given weight vector $w$, let $\hat w_g$ be its quantisation using grid $g$. The quality of a grid is measured by the mean‑squared error

$$
\mathrm{MSE}_g = \frac1n \sum_{i=1}^{n}\bigl(w_i-\hat w_{g,i}\bigr)^2 .
$$

The task is to pick the grid with the smallest $\mathrm{MSE}_g$.

## Task

Implement a function

```python
def best_grid(weights: np.ndarray) -> str:
    ...
```

that receives a one‑dimensional NumPy array of real weights and returns the string `"NF4"`, `"FP4"` or `"INT4"` corresponding to the grid that yields the lowest MSE when quantising `weights`. The function must be pure, use only NumPy vectorised operations and run in $O(n)$ time.

## Example

```python
import numpy as np

w = np.array([0.1, -0.5, 2.3, -7.9])
print(best_grid(w))
# → "FP4"
```

In this toy example the FP4 grid gives a smaller MSE than NF4 or INT4.

## What the gate checks

The grader generates a random normal weight vector of length $1000$, computes the exact MSE for each grid using NumPy, and records the label with the smallest error. Your implementation must return that same label; otherwise the `exact_match` metric fails.

In case of a tie the order **NF4 → FP4 → INT4** is used to break ties.
