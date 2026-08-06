## Context

Gaussian elimination factors a square matrix as $A = LU$, with $L$ unit
lower-triangular and $U$ upper-triangular. When a diagonal entry used as a
pivot is very small, the multiplier

$$m_{ik} = \frac{A_{ik}}{A_{kk}}$$

used to eliminate below it becomes huge. In exact arithmetic the
factorization is still correct, but in floating point the update

$$A_{ij} \leftarrow A_{ij} - m_{ik} A_{kj}$$

now adds a tiny number to a value that has been blown up by a factor of
$1/A_{kk}$, and the original, well-scaled entry $A_{ij}$ is rounded away
completely. This is why **partial pivoting** exists: at step $k$, swap in the
row from $\{k, \dots, n-1\}$ whose entry in column $k$ has the *largest
absolute value*, giving $PA = LU$ for a permutation matrix $P$.

The function below implements this factorization, but its pivot selection
has a bug: it only swaps rows when the current diagonal candidate is
*exactly* zero. A tiny-but-nonzero pivot is accepted as-is, so the matrix
below silently destroys precision instead of raising an error:

$$A = \begin{pmatrix} 10^{-12} & 1 \\ 1 & 1 \end{pmatrix}$$

Eliminating with pivot $A_{00} = 10^{-12}$ gives multiplier $m = 10^{12}$,
and $A_{11} - m \cdot A_{01} = 1 - 10^{12}$ rounds, in float64, to
$-10^{12}$ — the original $1$ is gone. Swapping rows first (pivot $=1$,
multiplier $=10^{-12}$) avoids the cancellation entirely.

## Task

Fix `lu_partial_pivot` so it always searches for the largest-magnitude
pivot candidate, not just the first nonzero one:

```python
def lu_partial_pivot(A: list[list[float]]) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    ...
```

- `A` — square list of lists of floats of shape $(n, n)$.
- Returns `(P, L, U)`:
  - `P` — $(n, n)$ permutation matrix (float64 entries, exactly one `1.0` per row and column) such that $PA = LU$.
  - `L` — $(n, n)$ **unit** lower-triangular matrix (ones on the diagonal, zeros strictly above it).
  - `U` — $(n, n)$ upper-triangular matrix (zeros strictly below the diagonal).


At elimination step $k$, the pivot row must be $\arg\max_{i \in \{k,\dots,n-1\}} \vert{}A_{ik}\vert{}$ (searched over the *current*, partially-eliminated matrix), not merely the first row with a nonzero entry. Remember to also swap the already-computed multipliers of $L$ in columns $< k$ when you swap rows, so that $PA = LU$ still holds exactly.

## Example

```python
A = [[1e-12, 1.0],
              [1.0,   1.0]]
P, L, U = lu_partial_pivot(A)
PA = [[sum(P[i][k] * A[k][j] for k in range(len(A))) for j in range(len(A))] for i in range(len(A))]
LU = [[sum(L[i][k] * U[k][j] for k in range(len(A))) for j in range(len(A))] for i in range(len(A))]
print(max(abs(PA[i][j] - LU[i][j]) for i in range(len(A)) for j in range(len(A))))   # ~1e-16 with correct pivoting
```

With the buggy "first nonzero pivot" version this reconstruction error is of order $1$, because the tiny pivot at $(0,0)$ was accepted without a swap.

## What the gate checks

A single gate named **max_abs_err** reconstructs $PA$ and $LU$ on several ill-scaled matrices (one fixed fixture plus freshly generated ones of different sizes and pivot magnitudes) and takes the worst-case $\max_{i,j} |(PA)_{ij} - (LU)_{ij}|$. It also checks that $L$ is unit lower-triangular, $U$ is upper-triangular, and $P$ is a genuine permutation matrix; violating any of these forces the gate to fail. The threshold is $10^{-8}$: correct partial pivoting reconstructs near machine precision ($\sim 10^{-15}$), while the "first nonzero pivot" bug produces errors around $10^{-4}$ on these matrices.
