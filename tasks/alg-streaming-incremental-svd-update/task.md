## Context

Singular value decomposition (SVD) factorizes a matrix $A \in \mathbb{R}^{m \times n}$ as

$$
A = U \Sigma V^\top ,
$$

where the diagonal entries of $\Sigma$ are the singular values. In PCA and
low-rank modeling, only the largest $k$ singular values and their associated
vectors are usually kept.

A streaming update receives a previous low-rank approximation and a new block
of rows. If the current approximation is

$$
A \approx U \Sigma V^\top ,
$$

and a new row block $X \in \mathbb{R}^{r \times n}$ arrives, the goal is to
produce new rank-$k$ factors approximating

$$
\begin{bmatrix}
U \Sigma V^\top \\
X
\end{bmatrix}.
$$

A practical incremental SVD method forms the reconstructed matrix, recomputes
the decomposition, and keeps the leading components. More optimized algorithms
avoid reconstructing the full matrix, but they must produce equivalent singular
values.

## Task

Implement `incremental_svd_update(U, S, Vt, X_new, k)`.

The arguments are:

- `U`: a 2-D NumPy array containing the left singular vectors of the current approximation.
- `S`: a 1-D NumPy array containing singular values.
- `Vt`: a 2-D NumPy array containing the transposed right singular vectors.
- `X_new`: a 2-D NumPy array containing newly arrived rows.
- `k`: the number of singular components to keep.

Return a tuple `(U_new, S_new, Vt_new)` representing a rank-$k$ SVD approximation
of the matrix formed by appending `X_new` below the current reconstruction.

The returned arrays must have compatible SVD shapes:

$$
U_{\text{new}} \in \mathbb{R}^{m' \times k}, \quad
S_{\text{new}} \in \mathbb{R}^{k}, \quad
Vt_{\text{new}} \in \mathbb{R}^{k \times n}.
$$

Use NumPy linear algebra operations.

## Example

```python
import numpy as np

U = np.eye(2)
S = np.array([3.0, 1.0])
Vt = np.eye(2)
X_new = np.array([[2.0, 0.0]])

U2, S2, Vt2 = incremental_svd_update(U, S, Vt, X_new, 2)

# S2 contains the two largest singular values of:
# [[3, 0],
#  [0, 1],
#  [2, 0]]
```

## What the gate checks

The grader builds several streaming update cases and compares the returned
singular values with a NumPy full SVD oracle. The reported metric is

$$
\mathrm{rel\_err} =
\frac{\lVert S_{\mathrm{candidate}} - S_{\mathrm{oracle}}\rVert_2}
{\lVert S_{\mathrm{oracle}}\rVert_2 + 10^{-12}} .
$$

The gate requires $\mathrm{rel\_err} \le 10^{-3}$.
