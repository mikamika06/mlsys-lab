## Context

For a symmetric matrix $A \in \mathbb{R}^{n \times n}$ and a unit vector $v$,
the Rayleigh quotient

$$
\mu(v) = \frac{v^\top A v}{v^\top v}
$$

is the best least-squares estimate of an eigenvalue given the direction $v$.
If $v$ is exactly an eigenvector, $\mu(v)$ is exactly the corresponding
eigenvalue.

Plain inverse iteration solves $(A - \sigma I) w = v$ for a *fixed* shift
$\sigma$ and repeats, converging linearly toward the eigenvalue nearest
$\sigma$. Rayleigh quotient iteration (RQI) instead updates the shift every
step, using the current Rayleigh quotient itself as the shift:

$$
\mu_k = \frac{v_k^\top A v_k}{v_k^\top v_k}, \qquad
w_{k+1} = (A - \mu_k I)^{-1} v_k, \qquad
v_{k+1} = \frac{w_{k+1}}{\lVert w_{k+1} \rVert} .
$$

Because the shift tracks the current best eigenvalue estimate, $A - \mu_k I$
becomes increasingly close to singular as $v_k$ approaches an eigenvector —
which is exactly what makes the *next* solve amplify the eigenvector
component so strongly. This adaptive shifting gives RQI **cubic**
convergence near a simple eigenvalue, versus the linear convergence of a
fixed-shift inverse iteration: once $v_k$ is reasonably close to an
eigenvector, only a handful of iterations are needed to reach machine
precision.

## Task

Implement `rayleigh_quotient_iteration(A, v0, n_iter)`:

```python
def rayleigh_quotient_iteration(A: np.ndarray, v0: np.ndarray, n_iter: int) -> float:
    ...
```

* `A` is a symmetric `float64` array of shape $(n, n)$.
* `v0` is a `float64` unit-norm starting vector of shape $(n,)$.
* `n_iter` is the number of RQI steps to perform.

Starting from $v_0$, run `n_iter` steps of the update above (compute the
Rayleigh quotient, solve the shifted system, renormalize) and return the
**final Rayleigh quotient** $\mu_{n\_iter}$ as a Python `float` — the
converged eigenvalue estimate. Use `np.linalg.solve` for the linear solve.

## Example

```python
import numpy as np
A = np.diag([1.0, 2.0, 5.0])
v0 = np.array([0.1, 0.05, 0.99])
v0 = v0 / np.linalg.norm(v0)
rayleigh_quotient_iteration(A, v0, 6)
# ~= 5.0  (v0 starts close to the eigenvector for eigenvalue 5)
```

## What the gate checks

The grader builds several random symmetric matrices with well-separated
eigenvalues (via `np.linalg.eigh` as the oracle), picks one target
eigenvector, and perturbs it slightly to form `v0` — so a correct RQI
implementation is guaranteed to converge to that specific target eigenvalue.
It runs your function for 8 iterations and compares the returned value to
the oracle's target eigenvalue with

$$
\mathrm{rel\_err} = \frac{|\hat{\lambda} - \lambda|}{|\lambda| + 10^{-12}} .
$$

The gate requires $\mathrm{rel\_err} \le 10^{-9}$ on every case. Thanks to
cubic convergence, a correct implementation reaches this accuracy well within
8 iterations; an implementation that uses a fixed shift, forgets to
renormalize, or does not update $\mu$ every step converges far too slowly (or
not at all) to pass.
