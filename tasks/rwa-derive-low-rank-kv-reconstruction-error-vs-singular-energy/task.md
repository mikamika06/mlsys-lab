## Context

In transformer attention, Key and Value matrices are often approximated by low-rank
factorisations to cut memory and compute. The quality of a rank-$r$ approximation
is governed by the singular-value spectrum of the matrix.

Let $A \in \mathbb{R}^{m \times n}$ have singular values
$s_1 \ge s_2 \ge \cdots \ge s_k \ge 0$, where $k = \min(m,n)$.
The Eckart–Young–Mirsky theorem guarantees that the best rank-$r$ approximation
$A_r$ (in both operator and Frobenius norm) is obtained by keeping the top $r$
singular values and zeroing out the rest.

The squared Frobenius reconstruction error is the sum of the discarded
singular-value energies:

$$\lVert A - A_r \rVert_F^2 \;=\; \sum_{i=r+1}^{k} s_i^{\,2}$$

The captured singular energy fraction at rank $r$ is:

$$\text{energy}(r) \;=\; \frac{\sum_{i=1}^{r} s_i^{\,2}}{\sum_{i=1}^{k} s_i^{\,2}}$$

These two quantities are complementary:

$$\lVert A - A_r \rVert_F^2 \;=\; \bigl(1 - \text{energy}(r)\bigr)\;\lVert A \rVert_F^2$$

In practice, the singular values decay rapidly for KV matrices arising from
natural-language sequences, so a small $r$ captures the vast majority of the
energy and the reconstruction error is near-zero.

## Task

Implement `kv_reconstruction_error_and_energy`:

```python
def kv_reconstruction_error_and_energy(matrix: np.ndarray, rank: int) -> tuple[float, float]:
    ...
```

Given a 2-D NumPy array `matrix` of shape $(m, n)$ and a target rank `rank` ($0 \le r \le k$), return a tuple `(reconstruction_error, energy_fraction)` where:

- `reconstruction_error` is $\lVert A - A_r \rVert_F^2$ (the **squared** Frobenius norm of the residual — not the Frobenius norm itself).
- `energy_fraction` is $\text{energy}(r)$ as defined above.

Use `np.linalg.svd` to decompose the matrix. Work directly with the singular values;
do not explicitly form $A_r$ and subtract.

## Example

```python
import numpy as np
A = np.array([[1.0, 0.0],
              [0.0, 2.0],
              [0.0, 0.0]])
# Singular values s = [2.0, 1.0], so s^2 = [4.0, 1.0]
err, energy = kv_reconstruction_error_and_energy(A, 1)
# err    = 1.0   (s_2^2 = 1.0)
# energy = 0.8   (s_1^2 / (s_1^2 + s_2^2) = 4/5)
```

## What the gate checks

The grader runs `np.linalg.svd` on seven test matrices (random tall, square,
wide, rank-deficient, full-rank, and rank-0) and independently computes both the
reconstruction error and energy fraction from the oracle singular values. It then
takes the maximum relative error over both quantities across all test cases. The
gate passes when this maximum is below $10^{-6}$.

A common mistake is returning the Frobenius norm $\lVert A - A_r \rVert_F$ instead
of its square, or using $\sum s_i / \sum s_j$ instead of $\sum s_i^2 / \sum s_j^2$
for the energy fraction.
