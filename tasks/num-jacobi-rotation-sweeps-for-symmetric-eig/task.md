## Context

The **cyclic Jacobi eigenvalue algorithm** finds all eigenvalues of a real
symmetric matrix $A$ by repeatedly applying plane (Givens) rotations that zero
out one off-diagonal entry at a time. A rotation $J(p,q,\theta)$ is the
identity matrix except for
$$
J_{pp}=J_{qq}=\cos\theta=c, \qquad J_{pq}=-J_{qp}=\sin\theta=s.
$$
Applying $A \leftarrow J^\top A J$ only changes rows/columns $p$ and $q$. To
make the new $A_{pq}=0$, choose
$$
\tau = \frac{A_{qq}-A_{pp}}{2A_{pq}}, \qquad
t = \frac{\operatorname{sign}(\tau)}{|\tau|+\sqrt{1+\tau^2}}, \qquad
c=\frac{1}{\sqrt{1+t^2}}, \qquad s = tc.
$$
A **sweep** applies this to every pair $p<q$ once. Because zeroing one entry
can slightly perturb previously-zeroed entries, repeated sweeps are needed;
the sum of squared off-diagonal entries is guaranteed to decrease monotonically
each sweep, so the algorithm converges and the diagonal of $A$ converges to the
eigenvalues of the original matrix.

## Task

Implement `jacobi_eigenvalues`:

```python
def jacobi_eigenvalues(A: np.ndarray, sweeps: int = 100, tol: float = 1e-12) -> np.ndarray:
    ...
```

* `A` — a symmetric 2-D array of shape $(n, n)$.
* `sweeps` — maximum number of cyclic sweeps to run.
* `tol` — stop early once $\sqrt{2\sum_{p<q}A_{pq}^2}$ drops below `tol`.

Return the $n$ eigenvalues of `A` as a 1-D `float64` array, **sorted
ascending**. Implement the rotation sweeps yourself with explicit loops over
`(p, q)` pairs — do not call `numpy.linalg.eigh` / `numpy.linalg.eigvalsh` (or
any other library eigensolver) anywhere in your solution; the grader detects
and rejects that.

## Example

```python
import numpy as np
A = np.array([[4.0, 1.0], [1.0, 3.0]])
jacobi_eigenvalues(A)
# -> array([2.38196601, 4.61803399])   # matches np.linalg.eigvalsh(A) sorted
```

## What the gate checks

`rel_err` — the grader builds several random symmetric matrices (sizes 2, 3,
5, 8) plus one fixed 3x3 case, runs `jacobi_eigenvalues`, sorts the result, and
compares it against `np.linalg.eigvalsh` with the global relative-L2-norm
scorer; the worst case across all matrices must satisfy `rel_err < 1e-8`. The
grader also runs your function under a call tracer that fails the gate
immediately (`rel_err = inf`) if `numpy.linalg.eigh` or
`numpy.linalg.eigvalsh` is ever invoked during the call, so the eigenvalues
must actually come from your own rotation sweeps.
