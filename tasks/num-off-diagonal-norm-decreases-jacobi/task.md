## Context

The classical Jacobi eigenvalue algorithm diagonalizes a symmetric matrix
$A$ by repeatedly applying plane rotations $J(p, q, \theta)$ that each zero
out exactly one off-diagonal pair $A_{pq} = A_{qp}$. A **sweep** visits every
pair $(p, q)$ with $p < q$ once. Zeroing one pair generally disturbs
entries that a *previous* rotation in the same sweep had already zeroed, so
convergence takes several sweeps — but each individual rotation is an
orthogonal similarity transform, so it never increases the total size of
the off-diagonal entries. A natural progress measure is the off-diagonal
Frobenius norm

$$
\operatorname{offdiag}(A) = \sqrt{\sum_{i \neq j} A_{ij}^2}\, ,
$$

and the key fact this task exercises numerically is that, sweep after
sweep, $\operatorname{offdiag}(A)$ decreases monotonically toward zero.

For a pivot pair $(p, q)$, the standard numerically stable rotation is:

$$
\theta = \frac{A_{qq} - A_{pp}}{2A_{pq}}, \qquad
t = \begin{cases} 1 & \theta = 0 \\ \dfrac{\operatorname{sign}(\theta)}{|\theta| + \sqrt{\theta^2+1}} & \theta \neq 0 \end{cases},
\qquad
c = \frac{1}{\sqrt{t^2+1}}, \qquad s = tc.
$$

The diagonal and off-diagonal entries update as

$$
A_{pp}' = A_{pp} - tA_{pq}, \qquad A_{qq}' = A_{qq} + tA_{pq}, \qquad A_{pq}' = A_{qp}' = 0,
$$

and every other entry in rows/columns $p, q$ updates as

$$
A_{ip}' = A_{pi}' = cA_{ip} - sA_{iq}, \qquad A_{iq}' = A_{qi}' = sA_{ip} + cA_{iq} \qquad (i \neq p, q).
$$

After several sweeps some pairs settle to a residual that is nonzero only
due to floating-point round-off (not exact `0.0`), and dividing by such a
residual would overflow $\theta$. So a robust implementation treats an
entry as already converged — and skips its rotation — once
$|A_{pq}| \le \varepsilon_{\text{mach}}\,(|A_{pp}| + |A_{qq}|)$.

## Task

Implement `jacobi_offdiag_norms`:

```python
def jacobi_offdiag_norms(A: np.ndarray, n_sweeps: int) -> np.ndarray:
    ...
```

* `A` — a symmetric 2-D NumPy array of shape $(n, n)$. Do not mutate it;
  operate on a copy.
* `n_sweeps` — number of full sweeps to run.

Each sweep visits pivot pairs $(p, q)$ with $0 \le p < q < n$ in row-major
nested order ($p$ outer loop, $q$ inner loop), applying exactly one
rotation per pair with the formula above (skipping pairs already below the
$\varepsilon_{\text{mach}}$ threshold given above).

Return a 1-D array of length `n_sweeps + 1`: `result[0]` is
$\operatorname{offdiag}(A)$ *before* any sweep, and `result[k]` for
$k = 1, \dots, \text{n\_sweeps}$ is $\operatorname{offdiag}(A)$ *after* the
$k$-th sweep.

## Example

```python
import numpy as np
A = np.array([[4.0, 1.0, 0.5],
              [1.0, 3.0, 0.2],
              [0.5, 0.2, 2.0]])
norms = jacobi_offdiag_norms(A, n_sweeps=6)
print(norms)
# a strictly-non-increasing sequence of 7 numbers, ending near 0
```

## What the gate checks

A single gate named **max_abs_err** runs your function on the fixture
matrix and on a second, independently generated symmetric matrix, for
`n_sweeps = 8`. For each case it verifies your returned array is
non-negative, non-increasing (within floating-point tolerance) sweep to
sweep, and ends below $10^{-9}$ — any violation forces the gate to `inf`.
It then compares your full per-sweep sequence, entry by entry, against a
reference implementation of the exact algorithm above, and takes the
worst-case $\max |{\text{yours} - \text{reference}}|$ over both matrices.
The threshold is $10^{-9}$.
