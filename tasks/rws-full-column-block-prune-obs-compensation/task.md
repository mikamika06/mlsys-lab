## Context

Optimal Brain Surgeon (OBS) style pruning uses curvature information to compensate for the effect of removing weights. For a linear layer with weight matrix $W \in \mathbb{R}^{m \times n}$ and calibration activations $X \in \mathbb{R}^{n \times k}$, the reconstruction objective is based on preserving

$$
WX .
$$

A Hessian approximation for the layer inputs is

$$
H = \frac{1}{k}XX^\top + \lambda I ,
$$

where $\lambda$ is a small damping term. The inverse Hessian estimate $H^{-1}$ describes how changes in one column affect the remaining columns.

The pruning importance of an individual weight $w_{ij}$ in column $j$ can be estimated as

$$
\frac{w_{ij}^{2}}{(H^{-1})_{jj}} .
$$

After selecting weights to remove, OBS compensates for the output change. If a weight in column $q$ is removed, the remaining columns receive a correction proportional to

$$
W_{:,r} \leftarrow W_{:,r} -
\frac{w_{iq}}{(H^{-1})_{qq}} (H^{-1})_{qr}
$$

for later columns $r$. This allows the pruned matrix to better preserve the layer output.

## Task

Implement `sparsegpt_prune(W, X, sparsity, block_size)`:

```python
def sparsegpt_prune(
    W: np.ndarray,
    X: np.ndarray,
    sparsity: float,
    block_size: int
) -> np.ndarray:
    ...
```

The function receives a weight matrix $W$ with shape $(m,n)$ and calibration activations $X$ with shape $(n,k)$. It must return a new `float64` weight matrix with the same shape.

Use the following algorithm:

1. Compute the damped Hessian approximation

$$
H = \frac{1}{k}XX^\top + 10^{-4}I .
$$

2. Compute $H^{-1}$ using NumPy linear algebra.

3. Divide the columns of $W$ into consecutive blocks of `block_size`.

4. For each block, compute the importance score
$$
s_{ij} = \frac{w_{ij}^{2}}{(H^{-1})_{jj}}
$$
for every weight in that block. Set the lowest `sparsity` fraction of weights in the block to zero.

5. Process columns from left to right. When a masked weight is removed, apply the OBS compensation update to all remaining columns in the same block and later blocks using the corresponding row of $H^{-1}$.

The implementation should use NumPy operations for matrix computations. The grading oracle uses the same mathematical procedure and checks the reconstructed layer output.

## Example

```python
import numpy as np

W = np.array([[1., 2., 3.],
              [4., 5., 6.]])
X = np.array([[1., 0., 1.],
              [0., 1., 1.],
              [1., 1., 0.]])

W_hat = sparsegpt_prune(W, X, 0.5, 2)

# W_hat is a pruned and compensated version of W.
# It preserves W_hat @ X close to W @ X.
```

## What the gate checks

The gate builds a deterministic NumPy reference implementation of the pruning and compensation algorithm. It compares the candidate output by measuring

$$
\mathrm{rel\_err} =
\frac{\lVert W_{\mathrm{candidate}}X -
W_{\mathrm{oracle}}X\rVert_F}
{\lVert W_{\mathrm{oracle}}X\rVert_F + 10^{-12}} .
$$

The returned `rel_err` must satisfy $\mathrm{rel\_err} \le 10^{-5}$.
