## Context

A matrix $A \in \mathbb{R}^{m \times n}$ can be approximated with a low-rank
factorization using the singular value decomposition (SVD):

$$
A = U \Sigma V^\top .
$$

Keeping only the largest $k$ singular values gives the rank-$k$ approximation

$$
A_k = U_k \Sigma_k V_k^\top .
$$

The compressed representation stores three smaller objects instead of the full
matrix: $U_k$, the first $k$ columns of $U$; the diagonal values of
$\Sigma_k$; and $V_k^\top$, the first $k$ rows of $V^\top$.

The storage reduction can be measured by comparing the bytes of the original
matrix with the bytes of the stored factors:

$$
\mathrm{size\_ratio}
=
\frac{\mathrm{bytes}(A)}
{\mathrm{bytes}(U_k)+\mathrm{bytes}(\Sigma_k)+\mathrm{bytes}(V_k^\top)} .
$$

The reconstruction error is measured by mean squared error:

$$
\mathrm{MSE}
=
\frac{1}{mn}\sum_{i=1}^{m}\sum_{j=1}^{n}(A_{ij}-(A_k)_{ij})^2 .
$$

## Task

Implement `compress_svd(A, k)` and `reconstruct_svd(U, S, Vt)`.

`compress_svd` takes a 2-D NumPy array with floating point values and an integer
rank $k$. It must return the compact rank-$k$ SVD factors:

```python
def compress_svd(A: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ...
```

The returned values must be:

- `U_k` with shape `(m, k)`,
- `S_k` with shape `(k,)`,
- `Vt_k` with shape `(k, n)`.

Use NumPy SVD rather than implementing an iterative decomposition.

`reconstruct_svd` must rebuild the approximation:

```python
def reconstruct_svd(U: np.ndarray, S: np.ndarray, Vt: np.ndarray) -> np.ndarray:
    ...
```

It should return $U \Sigma V^\top$ using the compact factors.

## Example

```python
import numpy as np

A = np.outer(np.arange(1, 6), np.arange(1, 5)).astype(float)

U, S, Vt = compress_svd(A, 1)
A_hat = reconstruct_svd(U, S, Vt)
```

`A_hat` is the rank-1 approximation of `A`, and the stored factors use fewer
bytes than the original matrix.

## What the gate checks

The gate computes the reference decomposition with NumPy's SVD oracle and checks
the submitted implementation on several low-rank matrices.

The `size_ratio` metric must be at least $3.0$, meaning the stored rank-$k$
factors must use substantially fewer bytes than the original matrix.

The `mse` metric compares the reconstruction against the input matrix. The mean
squared reconstruction error must be no greater than $10^{-6}$.
