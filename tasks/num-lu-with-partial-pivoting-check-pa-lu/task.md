## Context

LU decomposition factors a matrix $A \in \mathbb{R}^{n \times n}$ into a lower triangular matrix $L$ and an upper triangular matrix $U$.

With partial pivoting, rows are permuted before elimination. The decomposition is

$$
PA = LU,
$$

where $P$ is a permutation matrix containing the row swaps.

At elimination step $k$, partial pivoting selects the row with the largest magnitude entry in the current pivot column:

$$
p = \arg\max_{i \ge k} |A_{ik}|.
$$

The selected row is swapped into position $k$. The values below the pivot are stored as multipliers in $L$, and the remaining updated matrix forms $U$.

## Task

Implement `lu_partial_pivot(A)`:

```python
def lu_partial_pivot(A: list[list[float]]) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    ...
```

The function receives a square floating-point list and returns `(P, L, U)` satisfying:

$$
PA \approx LU.
$$

Requirements:

- `P` must be a permutation matrix.
- `L` must be lower triangular with diagonal entries equal to $1$.
- `U` must be upper triangular.
- Row exchanges must use partial pivoting.

## Example

```python

A = [[2., 1.],
              [4., 3.]]

P, L, U = lu_partial_pivot(A)

# P @ A is approximately equal to L @ U
```

## What the gate checks

The gate builds a reference decomposition using the same mathematical algorithm with a Python-based oracle implementation. It compares the returned $P$, $L$, and $U$ against the oracle result and also verifies the reconstruction.

The checked error is

$$
\max_{i,j}|(P_{\text{ref}})_{ij} - P_{ij}| +
\max_{i,j}|(L_{\text{ref}})_{ij} - L_{ij}| +
\max_{i,j}|(U_{\text{ref}})_{ij} - U_{ij}|.
$$

The final gate value must be less than $10^{-10}$. A decomposition that skips row swaps fails on matrices where the first pivot requires permutation.
