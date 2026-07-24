## Context

SparseGPT reconstructs a pruned linear layer by using second-order information from calibration data. For a weight matrix $W \in \mathbb{R}^{m \times d}$ and calibration activations $X \in \mathbb{R}^{d \times n}$, the local quadratic approximation uses the Hessian-like matrix

$$
H = 2XX^\top + \lambda I .
$$

The inverse curvature information allows Optimal Brain Surgeon (OBS) updates. Removing a weight is not treated independently: the remaining weights are adjusted to compensate for the output error.

For a column $j$, the OBS importance is proportional to

$$
s_j = \frac{w_j^2}{H^{-1}_{jj}},
$$

where $w_j$ is a weight value and $H^{-1}_{jj}$ is the corresponding inverse-Hessian diagonal element. When a weight is removed, surviving columns receive a correction derived from the inverse Hessian so that the layer output $WX$ changes as little as possible.

## Task

Implement `sparsegpt_layerwise(W, X, sparsity, lam)`:

```python
def sparsegpt_layerwise(W: np.ndarray, X: np.ndarray,
                        sparsity: float, lam: float):
    ...
```

The function receives a weight matrix `W` with shape $(m, d)$ and calibration activations `X` with shape $(d, n)$. It must return:

```python
W_hat, mask
```

where `W_hat` is the reconstructed sparse weight matrix and `mask` is a boolean array with the same shape as `W` indicating retained weights.

Use the OBS procedure:

1. Compute $H = 2XX^\top + \lambda I$.
2. Compute the inverse Hessian information using a Cholesky-based inverse.
3. Select weights to remove using OBS importance scores.
4. Apply OBS compensation updates to the surviving weights.

The target sparsity is the fraction of weights set to zero. Use only NumPy operations.

## Example

```python
import numpy as np

W = np.array([[2.0, 0.2], [0.1, 3.0]])
X = np.array([[1.0, 0.0], [0.0, 1.0]])

W_hat, mask = sparsegpt_layerwise(W, X, 0.5, 0.01)

# W_hat contains two retained weights and two pruned weights.
# mask marks the retained entries.
```

## What the gate checks

The gate builds a deterministic small layer and computes a NumPy OBS oracle using the same Hessian and OBS update equations. The returned reconstruction is compared by the output error

$$
\lVert WX-\hat{W}X\rVert_F .
$$

The candidate must match the oracle reconstruction within numerical tolerance. It must also outperform magnitude pruning at the same sparsity level. A magnitude-pruning implementation removes the smallest absolute weights without OBS compensation and is used as a baseline.
