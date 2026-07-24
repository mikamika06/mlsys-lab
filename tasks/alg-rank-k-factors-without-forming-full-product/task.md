## Context

A matrix $A \in \mathbb{R}^{m \times n}$ can be decomposed with singular value decomposition:

$$
A = U \Sigma V^\top .
$$

A rank-$k$ approximation keeps only the first $k$ singular components:

$$
A_k = U_k \Sigma_k V_k^\top .
$$

The factor matrices contain all information needed to represent this approximation:

- $U_k \in \mathbb{R}^{m \times k}$
- $\Sigma_k \in \mathbb{R}^{k}$
- $V_k^\top \in \mathbb{R}^{k \times n}$

Building $A_k$ itself requires allocating an $m \times n$ matrix. For large matrices this can dominate memory usage even though the final factors are much smaller.

## Task

Implement `rank_k_factors(A, k)`:

```python
def rank_k_factors(A: np.ndarray, k: int):
    ...
```

Return `(U_k, S_k, V_k)` where the values are obtained from the singular value decomposition of `A`:

- `U_k` contains the first $k$ columns of $U`.
- `S_k` contains the first $k$ singular values.
- `V_k` contains the first $k$ rows of $V^\top`.

Use NumPy linear algebra routines. Do not construct the reconstructed matrix
$U_k \Sigma_k V_k$.

## Example

```python
import numpy as np

A = np.array([
    [4.0, 0.0],
    [0.0, 1.0],
    [0.0, 0.0],
])

U_k, S_k, V_k = rank_k_factors(A, 1)

# S_k contains the largest singular value.
# U_k and V_k contain the corresponding singular vectors.
```

## What the gate checks

The gate computes reference factors using NumPy's SVD and compares the returned
factors after resolving the sign ambiguity of singular vectors.

The `factor_rel_err` metric must satisfy
$\mathrm{factor\_rel\_err} \le 10^{-6}$.

The gate also measures additional memory allocated while running the function
relative to the NumPy SVD reference. The `extra_alloc_bytes` metric must stay
below the limit. Implementations that materialize the full rank-$k$ product
allocate an unnecessary $m \times n$ result and fail this check.
