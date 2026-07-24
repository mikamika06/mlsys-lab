## Context

The QR decomposition factors a matrix $A \in \mathbb{R}^{m \times n}$ as

$$
A = QR,
$$

where the columns of $Q$ are orthonormal. A computed basis can lose
orthogonality because floating-point operations introduce rounding errors.

Classical Gram-Schmidt (CGS) computes each vector by subtracting all previous
projections:

$$
v_j = a_j - \sum_{i<j}(q_i^\top a_j)q_i .
$$

Modified Gram-Schmidt (MGS) performs the same projections but updates the
remaining columns after each new orthogonal vector is created. In exact
arithmetic both methods produce the same result, but on ill-conditioned
matrices MGS usually preserves orthogonality better.

The loss of orthogonality is measured as

$$
\lVert Q^\top Q - I \rVert_{\max},
$$

where $\lVert\cdot\rVert_{\max}$ is the largest absolute matrix entry.

## Task

Implement `gram_schmidt_orthogonality(A)`:

```python
def gram_schmidt_orthogonality(A: np.ndarray) -> tuple[float, float]:
    ...
```

The function receives a two-dimensional full-rank NumPy array and returns

```python
(cgs_loss, mgs_loss)
```

where the two values are the maximum absolute orthogonality errors produced by
Classical Gram-Schmidt and Modified Gram-Schmidt. Use `float64` arithmetic.

## Example

```python
import numpy as np

A = np.array([
    [1.0, 1.0, 1.0],
    [0.0, 1e-4, 0.0],
    [0.0, 0.0, 1e-4],
    [1.0, 1.0 + 1e-4, 1.0 - 1e-4],
])

cgs_loss, mgs_loss = gram_schmidt_orthogonality(A)
```

The returned values are non-negative floats. For ill-conditioned inputs,
`mgs_loss` should normally be smaller than `cgs_loss`.

## What the gate checks

The grader builds an ill-conditioned Hilbert matrix and computes an independent
NumPy reference implementation of both Gram-Schmidt variants.

The `max_abs_err` metric checks that both returned values match the oracle:

$$
\max(|x_{\mathrm{student}} - x_{\mathrm{reference}}|) \le 10^{-12}.
$$

The `mgs_orthogonality` metric checks that the Modified Gram-Schmidt result
satisfies

$$
\lVert Q^\top Q - I \rVert_{\max} < 10^{-8}.
$$
