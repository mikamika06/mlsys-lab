## Context

The singular value decomposition of a matrix $A \in \mathbb{R}^{m \times n}$ is

$$
A = U \Sigma V^\top ,
$$

where the diagonal entries of $\Sigma$ are the singular values
$\sigma_1 \ge \sigma_2 \ge \dots \ge 0$.

The best rank-$k$ approximation in the Frobenius norm is the truncated SVD:

$$
A_k = U_k \Sigma_k V_k^\top .
$$

The Eckart-Young theorem states that the reconstruction error is determined by the
first discarded singular value:

$$
\lVert A - A_k \rVert_2 = \sigma_{k+1}.
$$

Therefore, the spectral reconstruction error of the best rank-$k$ approximation can
be reported without explicitly constructing $A_k$ by reading the next singular
value after the retained components.

## Task

Implement `truncated_rank_k_error(A, k)`:

```python
def truncated_rank_k_error(A: np.ndarray, k: int) -> float:
    ...
```

The function takes a two-dimensional NumPy array $A$ and an integer $k$ with
$0 < k < \min(A.shape)$. Return the spectral reconstruction error of the best
rank-$k$ approximation.

Use SVD-based reasoning. The returned value must be a Python `float`.

## Example

```python
import numpy as np

A = np.array([
    [3.0, 0.0],
    [0.0, 2.0],
    [0.0, 1.0],
])

err = truncated_rank_k_error(A, 1)
# err is 2.0 because the discarded singular values are 2.0 and 1.0,
# and the largest discarded singular value is sigma_2 = 2.0.
```

## What the gate checks

The gate computes the oracle answer by running NumPy's SVD implementation and
taking the first discarded singular value $\sigma_{k+1}$. The submitted function
is tested on several matrices and values of $k$.

The relative error between the submitted reconstruction error and the NumPy
oracle value must satisfy

$$
\mathrm{rel\_err} =
\frac{|x-\hat{x}|}{|x|+10^{-12}} < 10^{-6}.
$$

A function that returns an incorrect singular value or uses the wrong rank
convention will fail.
