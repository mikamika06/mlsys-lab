## Context

PCA on a large covariance matrix rarely needs the full eigendecomposition — only the
top $k$ directions. Plain power iteration finds one eigenvector at a time, so getting
$k$ of them means **deflation**: run the iteration, subtract the found direction, run
again. That is $k$ sequential Python passes, each with an inner Gram-Schmidt loop over
the already-found vectors.

*Subspace iteration* (also called block power iteration, or orthogonal iteration) does
all $k$ directions at once. Start from a block $Q_0 \in \mathbb{R}^{n \times k}$ and
repeat

$$
Z_{t} = A\,Q_{t-1}, \qquad Q_{t} R_{t} = Z_{t} \quad (\text{thin QR}),
$$

so every iteration is one matrix product plus one QR factorisation. Both are single
BLAS/LAPACK calls — the whole Python loop body is $O(1)$ lines regardless of $k$.

After the block has converged, the individual eigenpairs come from a **Rayleigh-Ritz**
projection onto the converged subspace:

$$
M = Q^{\mathsf T} A\, Q \in \mathbb{R}^{k \times k}, \qquad
M = V \Lambda V^{\mathsf T}, \qquad
\hat{Q} = Q V .
$$

The diagonal of $\Lambda$ holds the Ritz values (approximate eigenvalues) and the
columns of $\hat{Q}$ the Ritz vectors. Convergence of the subspace is governed by the
spectral gap: the error decays like $(\lambda_{k+1}/\lambda_k)^{t}$.

## Task

Implement `block_power_topk` in `solve.py`:

```python
def block_power_topk(A: np.ndarray, Q0: np.ndarray, n_iter: int):
    ...
```

* `A` — symmetric matrix of shape $(n, n)$.
* `Q0` — starting block of shape $(n, k)$, **not** orthonormal.
* `n_iter` — number of block iterations.

Return a tuple `(eigvals, Q)`:

* `eigvals` — shape $(k,)$, `float64`, the top-$k$ eigenvalues of $A$ sorted in
  **descending** order.
* `Q` — shape $(n, k)$, `float64`, with column $j$ a unit-norm eigenvector for
  `eigvals[j]`. The overall sign of a column is free (the grader aligns signs).

Do the orthonormalisation **on the whole block at once** (one thin QR per iteration).
Do not deflate one vector at a time with a Python loop over the $k$ columns.

## Example

```python
import numpy as np

A = np.diag([4.0, 2.0, 0.5])
Q0 = np.array([[1.0, 0.0],
               [0.3, 1.0],
               [0.1, 0.2]])

eigvals, Q = block_power_topk(A, Q0, 40)

# eigvals -> [4.0, 2.0]
# Q       -> columns ~ [1,0,0] and [0,1,0]  (up to sign)
```

## What the gate checks

The grader loads the hidden matrix `A` and start block `Q0` from fixtures, and also
builds a second problem of its own. For each one it computes the oracle with
`numpy.linalg.eigh` and keeps the top-$k$ eigenpairs. Nothing is hardcoded.

Three accuracy metrics, all the relative $L_2$ error
$\lVert \hat{x} - x \rVert_2 / \lVert x \rVert_2$:

| metric | compares |
| --- | --- |
| `eigval_rel_err` | your `eigvals` vs the oracle eigenvalues |
| `component_rel_err` | your `Q` (column signs aligned to the oracle) vs the oracle eigenvectors |
| `subspace_rel_err` | the projector $QQ^{\mathsf T}$ vs $V V^{\mathsf T}$ — sign- and rotation-free |

All three must be $\leq 10^{-4}$.

A fourth gate, `line_events`, counts Python-level line executions during one call using
`sys.settrace` (the function is called once un-traced first, so lazy imports are not
counted). A blocked implementation stays around a few thousand events; a per-vector
deflation loop with an inner Gram-Schmidt pass costs roughly $k$ times more and blows
past the budget of **16000**. Vectorised NumPy work is invisible to the tracer, so the
gate measures Python-loop structure only, not machine speed.
