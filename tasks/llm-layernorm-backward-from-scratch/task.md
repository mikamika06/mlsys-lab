## Context

Layer Normalization (LayerNorm) normalizes each sample across its feature dimension.  
For an input matrix $X \in \mathbb{R}^{N\times D}$ the forward pass is

$$\mu_i = \frac1D\sum_{j=1}^D X_{ij}, \qquad
\sigma_i^2 = \frac1D\sum_{j=1}^D (X_{ij}-\mu_i)^2 + \varepsilon,$$

$$\hat{X}_{ij} = \frac{X_{ij}-\mu_i}{\sqrt{\sigma_i^2}}, \qquad
Y_{ij} = \gamma_j\,\hat{X}_{ij} + \beta_j.$$

Here $\gamma,\beta \in \mathbb{R}^{D}$ are learnable parameters and $\varepsilon>0$ is a small constant for numerical stability.

The gradient of a scalar loss $L$ with respect to the input $X$ can be derived analytically.  
Let $dY = \partial L/\partial Y$.  The per‑sample derivative is

$$
\frac{\partial L}{\partial X_{ij}}
= \gamma_j\,\sigma_i^{-1}\!\left(
dY_{ij}
- \frac{1}{D}\sum_{k} dY_{ik}
- \hat{X}_{ij}\,\frac{1}{D}\sum_{k} dY_{ik}\hat{X}_{ik}
\right).
$$

This expression is fully vectorizable with NumPy broadcasting.

## Task

Implement the function

```python
def compute_dx(dy: np.ndarray,
               x: np.ndarray,
               gamma: np.ndarray,
               beta: np.ndarray,
               eps: float = 1e-5) -> np.ndarray:
    ...
```

It receives:

* `dy` – gradient of a loss with respect to the LayerNorm output, shape `(N,D)`.
* `x`   – original input to the forward pass, shape `(N,D)`.
* `gamma`, `beta` – parameters used in the forward pass.
* `eps` – numerical stability constant (default `1e-5`).

Return `dx`, the gradient of the loss with respect to `x`.  
The implementation must use only NumPy vectorized operations; no explicit Python loops over samples or features.

## Example

```python
import numpy as np
from compute_dx import compute_dx

N, D = 2, 3
x      = np.array([[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0]])
gamma  = np.array([1.0, 0.5, -0.5])
beta   = np.zeros(D)
dy     = np.ones((N,D))

dx = compute_dx(dy, x, gamma, beta)
print(dx)
```

The output is a `(2,3)` array of gradients computed from the analytic formula.

## What the gate checks

* **Relative error** – The returned `dx` must match the gradient obtained by central finite differences applied to the forward LayerNorm implementation.  
  The metric `rel_err = ||dx - dx_ref|| / ||dx_ref||` is required to be ≤ $10^{-4}$.
* **Pure NumPy** – The solution should not contain explicit Python loops over array elements; it must rely on broadcasting and vectorized operations.

The grader computes the reference gradient by perturbing each element of `x` with a small step size, evaluating the forward pass, and forming the loss $L = \sum_{i,j} dy_{ij}\,Y_{ij}$.
