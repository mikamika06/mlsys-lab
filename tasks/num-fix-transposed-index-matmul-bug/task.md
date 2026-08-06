## Context

Matrix multiplication of $A \in \mathbb{R}^{m \times k}$ and $B \in \mathbb{R}^{k \times n}$
produces $C \in \mathbb{R}^{m \times n}$ where every entry is a dot product of a row of
$A$ with a column of $B$:

$$
C_{ij} = \sum_{p=0}^{k-1} A_{ip}\, B_{pj}, \qquad i = 0,\dots,m-1,\; j = 0,\dots,n-1.
$$

Writing this out as three nested loops, the inner accumulation reads element
$p$ of row $i$ from $A$ and element $p$ of column $j$ from $B$ — that second
read must index $B$ as `B[p, j]`, *not* `B[j, p]`. Swapping those two indices
silently turns the multiply into the product with $B^\top$ instead of $B$,
which only produces the right answer when $B$ happens to be square and
symmetric.

## Task

A colleague's pure-Python triple-loop implementation of matrix multiplication
lives in `starter.py`. It reads `B[j, p]` where it should read `B[p, j]`, so
the result is actually `A @ B.T` (or, when the shapes don't even allow that,
it raises an `IndexError`). Find the bug and fix it.

The function signature is:

```python
def matmul_naive(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """Compute A @ B for A of shape (m,k) and B of shape (k,n) using explicit
    Python loops. Returns a float64 list of shape (m,n)."""
```

`A` and `B` are list. `A` has shape $(m,k)$ and `B` has shape $(k,n)$;
in general $k \neq n$, so a correct fix must actually swap the index order,
not just happen to work by accident.

## Example

```python
A = [[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0]]            # shape (2, 3)
B = [[ 7.0,  8.0],
              [ 9.0, 10.0],
              [11.0, 12.0]]               # shape (3, 2)

C = matmul_naive(A, B)
# C == [[ 58.0,  64.0],
#       [139.0, 154.0]]
```

## What the gate checks

The gate computes the maximum absolute error

$$
\mathrm{max\_abs\_err} = \max_{i,j} \left| C_{ij} - (A B)_{ij} \right|
$$

against `A @ B` computed by Python, on several random rectangular matrices
(including non-square cases where the buggy indexing raises an `IndexError`
or produces a completely wrong shape). The maximum error over all cases must
satisfy $\mathrm{max\_abs\_err} \le 10^{-6}$.
