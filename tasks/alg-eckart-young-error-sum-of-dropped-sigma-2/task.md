## Context

The Eckart-Young theorem describes the optimal rank-$k$ approximation of a matrix. For a matrix $X \in \mathbb{R}^{m \times n}$ with singular values

$$
\sigma_1 \geq \sigma_2 \geq \dots \geq \sigma_r \geq 0,
$$

where $r = \min(m,n)$, the best rank-$k$ approximation $X_k$ is obtained by keeping the first $k$ singular components of the singular value decomposition.

The theorem gives the reconstruction error:

$$
\lVert X - X_k \rVert_F^2 = \sum_{i=k+1}^{r} \sigma_i^2 .
$$

This means the squared Frobenius error after dropping singular values can be computed either by explicitly reconstructing the truncated matrix or by summing the squared singular values that were removed.

## Task

Implement `eckart_young_errors(X)`:

```python
def eckart_young_errors(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ...
```

The input is a 2-D NumPy array `X` with real-valued entries.

Return two one-dimensional arrays of length $r+1$, where $r = \min(m,n)$.

The first array contains the direct reconstruction errors:

$$
E_k = \lVert X - X_k \rVert_F^2
$$

for every $k$ from $0$ through $r$.

The second array contains the Eckart-Young prediction:

$$
S_k = \sum_{i=k+1}^{r} \sigma_i^2 .
$$

Use NumPy linear algebra operations. The returned arrays must have dtype `float64`.

## Example

```python
import numpy as np

X = np.array([
    [3.0, 0.0],
    [0.0, 1.0],
])

direct, theorem = eckart_young_errors(X)

# Both arrays describe the same errors:
# direct  -> [10.0, 1.0, 0.0]
# theorem -> [10.0, 1.0, 0.0]
```

## What the gate checks

The gate computes a NumPy SVD reference for several matrices. It compares the returned error curves against the oracle values using mean squared error:

$$
\mathrm{MSE} = \frac{1}{N}\sum_i (y_i-\hat{y}_i)^2 .
$$

The combined output must have $\mathrm{MSE} \leq 10^{-8}$. Solutions that confuse singular values with singular-value squares or use only one of the two required curves will fail.
