## Context

Weight pruning methods remove columns from a weight matrix while attempting to preserve the effect of the removed weights. A reconstruction step propagates the error from a masked column into columns that have not yet been pruned.

Given weights $W \in \mathbb{R}^{m \times n}$ and calibration activations $X \in \mathbb{R}^{n \times s}$, define the damped curvature matrix

$$
H = XX^\top + \lambda I ,
$$

where $\lambda = 10^{-6}$ and $I$ is the identity matrix. The inverse curvature matrix $H^{-1}$ determines how masking error is distributed.

When column $q$ is removed, its current value is the reconstruction error. Every not-yet-removed column $j$ receives the correction

$$
\Delta W_j = W_q \frac{(H^{-1})_{qj}}{(H^{-1})_{qq}} .
$$

The updated weights are then used for the next pruning step. Tracking the intermediate states is useful because the correction changes the weights that will be removed later.

## Task

Implement `reconstruct_pruned_weights(W, X, prune_order)`:

```python
def reconstruct_pruned_weights(W: np.ndarray, X: np.ndarray, prune_order: list[int]) -> np.ndarray:
    ...
```

The function returns an array of shape $(k,m,n)$, where $k$ is the number of columns in `prune_order`. Entry `result[t]` must be the full compensated weight matrix immediately after pruning `prune_order[t]`.

Use this algorithm:

1. Copy $W$ into a working `float64` matrix.
2. Compute

$$
H = XX^\top + 10^{-6}I .
$$

3. Compute $H^{-1}$ through a Cholesky factorization and triangular solves.
4. For every position $t$ and column $q = \text{prune\_order}[t]$:
   - Save the current column $W_q$.
   - Set column $q$ to zero.
   - For every later column $j$ in `prune_order`, apply

$$
W_j \leftarrow W_j + W_q \frac{(H^{-1})_{qj}}{(H^{-1})_{qq}} .
$$

   - Store a copy of the resulting matrix in the output sequence.

Return the complete sequence of intermediate matrices.

## Example

```python
import numpy as np

W = np.array([[1., 2., 3.], [4., 5., 6.]])
X = np.array([[1., 0.], [0., 1.], [1., 1.]])

states = reconstruct_pruned_weights(W, X, [0, 2, 1])
print(states.shape)
# (3, 2, 3)
```

`states[0]` contains the matrix after column `0` is masked and its error is propagated into columns `2` and `1`.

## What the gate checks

The gate independently recomputes the Cholesky-based reconstruction sequence using a NumPy oracle.

The maximum absolute error

$$
\max_i |A_i - B_i|
$$

between the submitted sequence and the oracle sequence must be at most $10^{-6}$.

A solution that only zeros columns, ignores curvature correction, or returns only the final matrix will fail.
