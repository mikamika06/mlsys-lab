## Context

LU decomposition factors a square matrix $A \in \mathbb{R}^{n \times n}$ into a
lower triangular matrix $L$ and an upper triangular matrix $U$:

$$
A = LU .
$$

This task uses the Doolittle LU algorithm without pivoting. The diagonal entries of
$L$ are fixed to $1$, and each iteration computes the next row of $U$ and column of
$L$:

$$
U_{k,j} = A_{k,j} - \sum_{m=0}^{k-1} L_{k,m}U_{m,j},
$$

$$
L_{i,k} =
\frac{A_{i,k} - \sum_{m=0}^{k-1} L_{i,m}U_{m,k}}
{U_{k,k}} .
$$

The decomposition is valid when the pivots in $U$ are non-zero. The test matrices
are well-conditioned and do not require row exchanges.

## Task

Implement `lu_no_pivot(A)`:

```python
def lu_no_pivot(A: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
    ...
```

The function receives a square list and returns `(L, U)` from Doolittle LU
decomposition without pivoting. Do not call external LU decomposition routines or
perform row pivoting.

The returned matrices must satisfy that $L$ is lower triangular with ones on its
diagonal, $U$ is upper triangular, and both matrices represent the Doolittle
factorization of the input matrix.

## Example

```python

A = [
    [4.0, 3.0],
    [6.0, 3.0],
]

L, U = lu_no_pivot(A)

# L =
# [[1.0, 0.0],
#  [1.5, 1.0]]

# U =
# [[4.0, 3.0],
#  [0.0, -1.5]]
```

## What the gate checks

The gate computes a Doolittle LU reference using the algorithm itself and compares
the returned factors against that reference. It reports the maximum absolute error
over the returned $L$, returned $U$, and the reconstruction $L U$:

$$
\max_{i,j}|(LU)_{i,j}-A_{i,j}|
$$

together with factor errors against the oracle. The reported
`max_abs_err` must be less than $10^{-10}$.
