## Context

In Principal Component Analysis (PCA) we decompose a data matrix $A \in \mathbb{R}^{m\times n}$ into singular vectors and values
$$
A = U\Sigma V^\top,
$$
where $\Sigma=\operatorname{diag}(\sigma_1,\dots,\sigma_{\min(m,n)})$ contains the non‑negative singular values sorted in descending order. The squared singular values $\sigma_i^2$ are proportional to the variance captured by each principal component. If we keep only the first $k$ components, the retained energy is
$$
E_k = \frac{\sum_{i=1}^k\sigma_i^2}{\sum_{i=1}^{\min(m,n)}\sigma_i^2}.
$$

A common requirement is to choose the smallest $k$ such that $E_k \geq 0.9$, i.e., at least 90 % of the total variance is preserved.

When we store the truncated PCA model we need the matrices
$U_{m\times k}$, $\Sigma_{k\times k}$ (or just its diagonal), and $V^\top_{k\times n}$. The number of scalar values required is therefore
$$
N_{\text{comp}} = m\,k + k + k\,n.
$$
The footprint ratio relative to the original data matrix ($m\,n$ scalars) is
$$
\operatorname{size\_ratio}=\frac{m\,k+k+k\,n}{m\,n}.
$$

## Task

Implement `pick_rank_and_report(A)`:

```python
def pick_rank_and_report(A: np.ndarray) -> tuple[int, float]:
    ...
```

The function receives a 2‑D NumPy array $A$ of shape $(m,n)$ and must return a tuple `(k, size_ratio)` where

* `k` is the smallest integer such that the retained energy $E_k \geq 0.9$,
* `size_ratio` is the compression footprint ratio defined above.

The implementation must use only NumPy operations; no explicit Python loops are allowed.

## Example

```python
import numpy as np
A = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
k, size_ratio = pick_rank_and_report(A)
print(k)          # e.g. 2
print(size_ratio) # e.g. 0.666...
```

## What the gate checks

Two gates are applied:

* `exact_match` – verifies that the returned rank `k` equals the reference value.
* `rel_err` – computes the relative error of `size_ratio` against a NumPy reference and requires it to be at most $10^{-9}$.

Both metrics are computed by the grader using an exact SVD on random test matrices. A correct implementation will satisfy both gates.
