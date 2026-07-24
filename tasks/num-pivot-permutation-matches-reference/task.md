## Context

LU decomposition with **partial pivoting** factors $PA = LU$: at each elimination
step $k$, the row with the largest absolute value in the current (already
partially eliminated) column $k$ is swapped into position $k$ before
eliminating below it. LAPACK — and therefore `scipy.linalg.lu_factor` — does
not report $P$ as a full permutation matrix. Instead it reports a compact
**swap vector** `piv` of length $n$: `piv[k]` is the row index ($\geq k$) that
row $k$ was exchanged with *at step $k$*, applied to the matrix *as it stood
at that step* (i.e. `piv[k] == k` means no swap happened at step $k$). Row
indices below $k$ are never touched again, so re-applying the swaps
`for k in range(n): swap(row[k], row[piv[k]])` in order reconstructs $P$.

For a step $k$, the pivot row is chosen from the **Schur complement** — the
column $k$ of the matrix *after* the first $k$ elimination steps have already
been applied — not from the original matrix's column $k$.

## Task

Implement `lu_pivot_indices`:

```python
def lu_pivot_indices(A: np.ndarray) -> np.ndarray:
    ...
```

Given a square 2-D array `A` of shape $(n, n)$, run Gaussian elimination with
partial pivoting yourself (row swap + eliminate below the pivot at every step,
so later pivot choices see the correctly updated Schur complement) and return
the length-$n$ integer `piv` swap vector in LAPACK's convention described
above. `piv[n-1]` is always `n-1` (no row left to swap the last pivot with).
Do not call `scipy.linalg.lu_factor` / `scipy.linalg.lu` / `numpy.linalg`
LU-family routines — implement the elimination loop yourself.

## Example

```python
import numpy as np
A = np.array([[2.0, 1.0, 1.0],
              [4.0, 3.0, 3.0],
              [8.0, 7.0, 9.0]])
lu_pivot_indices(A)
# -> array([2, 2, 2])   # matches scipy.linalg.lu_factor(A)[1]
```

## What the gate checks

`exact_match` — the grader builds a fixed 3x3 case plus 40 random matrices
(sizes 2 through 8, entries drawn continuously so max-abs pivots are always
unambiguous — no ties) and compares your `piv` array, element for element, to
the real `scipy.linalg.lu_factor` oracle's `piv` output. All must match
exactly for the gate (`== 1.0`) to pass; a single differing entry anywhere
fails the whole case.
