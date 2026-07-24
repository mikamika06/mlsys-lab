## Context

Unstructured pruning removes individual weights from a dense matrix while keeping the remaining values. A common production representation is Compressed Sparse Row (CSR), which stores only nonzero values.

For a sparse weight matrix $W \in \mathbb{R}^{m \times k}$ and an input matrix $X \in \mathbb{R}^{k \times n}$, the dense operation is

$$Y = WX.$$

CSR stores $W$ using three arrays:

- `data` contains the nonzero values row by row.
- `indices` contains the column index of each value in `data`.
- `indptr` contains row boundaries, where row $i$ uses entries from `indptr[i]$ to `indptr[i+1]$.

A CSR implementation computes each output row by visiting only stored weights:

$$Y_{ij} = \sum_{p=indptr[i]}^{indptr[i+1]-1} data_p X_{indices_p,j}.$$

This is equivalent to multiplying the dense matrix by a binary sparsity mask $M$:

$$Y = (W \odot M)X,$$

where $\odot$ is elementwise multiplication.

## Task

Implement `csr_matmul(data, indices, indptr, X)`:

```python
def csr_matmul(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    X: np.ndarray,
) -> np.ndarray:
    ...
```

The arguments describe a CSR matrix with shape $(m, k)$:

- `data.shape[0]` is the number of stored weights.
- `indices` has the same length as `data` and stores column indices.
- `indptr` has length $m+1$ and stores row boundaries.
- `X` has shape $(k, n)$.

Return the dense result with shape $(m, n)$ and dtype `float64`.

Use the CSR representation directly. Do not reconstruct the full dense matrix.

## Example

```python
import numpy as np

data = np.array([2.0, 3.0, 4.0])
indices = np.array([0, 2, 1])
indptr = np.array([0, 2, 3])

X = np.array([
    [1.0, 2.0],
    [3.0, 4.0],
    [5.0, 6.0],
])

Y = csr_matmul(data, indices, indptr, X)

# The CSR matrix is:
# [[2, 0, 3],
#  [0, 4, 0]]
#
# Y is:
# [[17, 22],
#  [12, 16]]
```

## What the gate checks

The grader builds CSR matrices with unstructured nonzero locations and computes the oracle result by expanding the CSR matrix into a NumPy dense matrix and multiplying it with `X`.

The returned matrix is compared with the NumPy oracle using

$$\max_{i,j}|Y_{ij}^{candidate}-Y_{ij}^{oracle}|.$$

The gate requires this maximum absolute error to be less than $10^{-6}$.
