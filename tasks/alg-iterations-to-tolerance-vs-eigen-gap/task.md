## Context

The **power iteration** algorithm estimates the dominant eigenvector of a real symmetric matrix $A \in \mathbb{R}^{n\times n}$. Starting from an initial vector $\mathbf v_0$, it repeatedly applies
$$\mathbf w_k = A\,\mathbf v_{k-1}, \qquad \mathbf v_k = \frac{\mathbf w_k}{\lVert \mathbf w_k\rVert}.$$
If $A$ has eigenvalues $\lambda_1 > \lambda_2 \ge \dots \ge \lambda_n$, the sequence $\mathbf v_k$ converges to the eigenvector associated with $\lambda_1$. The speed of convergence is governed by the **eigen‑gap**
$$\gamma = 1 - \frac{\lambda_2}{\lambda_1}.$$
A larger gap yields faster convergence.

The stopping criterion we use here compares successive iterates:
$$\lVert \mathbf v_k - \mathbf v_{k-1}\rVert < \text{tol},$$
where $\text{tol}$ is a user‑supplied tolerance. The task is to return the number of iterations required to satisfy this condition.

## Task

Implement `iterations_to_tolerance(A, tol=1e-6)`:

```python
def iterations_to_tolerance(A: np.ndarray, tol: float = 1e-6) -> int:
    ...
```

`A` is a real symmetric NumPy array of shape `(n,n)`. The function must start from the vector of all ones (normalised to unit length), then perform power iteration until the Euclidean distance between two consecutive iterates falls below `tol`. Return the integer count of iterations performed. If convergence is not achieved within 10 000 steps, return `10000`.

The result must be an `int` and the algorithm must use only NumPy operations (no explicit Python loops over matrix entries).

## Example

```python
import numpy as np
A = np.diag([5., 3., 2.])
cnt = iterations_to_tolerance(A, tol=1e-8)
print(cnt)   # → 4
```

## What the gate checks

The grader evaluates your implementation against a reference that uses the same deterministic algorithm (initial vector of ones). For each fixture it compares the integer iteration count returned by your function to the oracle’s value. The metric `exact_match` must be `1.0`; any mismatch yields failure.
