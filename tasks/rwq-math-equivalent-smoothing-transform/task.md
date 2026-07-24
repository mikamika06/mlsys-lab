## Context

Let $X \in \mathbb{R}^{n\times d}$ and $W \in \mathbb{R}^{d\times m}$.  
For a non‑zero vector $s \in \mathbb{R}^d$ define the diagonal matrix $\operatorname{diag}(s)$.  
The following scaling transforms preserve the product:

$$
X' = X\,\operatorname{diag}\!\left(\frac{1}{s}\right), \qquad
W' = \operatorname{diag}(s)\,W .
$$

Indeed,

$$
X'\,W' = X\,\operatorname{diag}\!\left(\frac{1}{s}\right)
        \,\operatorname{diag}(s)\,W
      = X\,I\,W
      = X\,W .
$$

Thus $X'$ and $W'$ are *math‑equivalent* to the original pair $(X,W)$.

## Task

Implement a function that applies this smoothing transform:

```python
def smoothing_transform(X: np.ndarray, W: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ...
```

The returned matrices must satisfy:
- `X' @ W'` equals `X @ W` up to a very small numerical error.
- The shapes of `X'` and `W'` match those of `X` and `W`.
- All computations use `float64`.

## Example

```python
import numpy as np

X = np.array([[1, 2], [3, 4]], dtype=np.float64)
W = np.array([[5, 6], [7, 8]], dtype=np.float64)
s = np.array([2.0, 0.5])

Xp, Wp = smoothing_transform(X, W, s)

print("X' @ W':")
print(Xp @ Wp)          # [[19. 22.]
                        #  [43. 50.]]

print("X @ W:")
print(X @ W)            # same matrix
```

## What the gate checks

The grader computes the reference product `X @ W` and compares it with the candidate's `X' @ W'`.  
It reports the maximum absolute error:

$$
\max_{i,j} |(X'W')_{ij} - (XW)_{ij}|.
$$

A solution passes if this value is at most $10^{-5}$.

The grader also verifies that the returned matrices have the same shapes as the inputs and are of type `float64`. A failure in any of these checks results in a gate failure.
