## Context

Given a calibration activation matrix $X \in \mathbb{R}^{n\times d}$ produced by a linear layer, the per‑input (per‑column) $\ell_2$ norm is defined as
$$\mathbf{z}_j = \lVert X_{\:,j}\rVert_2 = \sqrt{\sum_{i=1}^n X_{ij}^2}, \qquad j=1,\dots,d.$$
These norms are used by Wanda to normalise activations before quantisation.

## Task

Implement `compute_activation_norms(X)`:

```python
def compute_activation_norms(X: np.ndarray) -> np.ndarray:
    ...
```

It receives a 2‑D NumPy array of shape $(n,d)$ and must return a 1‑D float64 array of length $d$ containing the columnwise $\ell_2$ norms. No explicit Python loops are allowed; use vectorised NumPy operations only.

## Example

```python
import numpy as np
X = np.array([[0, 3], [4, 0]], dtype=np.float32)
z = compute_activation_norms(X)
# z == array([5., 3.])   # shape (2,)
```

## What the gate checks

The grader computes a NumPy reference $\mathbf{z}_{\text{ref}}=\lVert X\rVert_2$ with `np.linalg.norm(X,axis=0)` and evaluates the global relative error
$$\mathrm{rel\_err} = \frac{\|\mathbf{z}-\mathbf{z}_{\text{ref}}\|}{\|\mathbf{z}_{\text{ref}}\|}.$$
The solution must achieve $\mathrm{rel\_err}\le 10^{-6}$ on several random test matrices.
