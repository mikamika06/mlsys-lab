## Context

LayerNorm normalizes each row of an activation matrix and then applies a
learned per-feature affine transform. For a row $x \in \mathbb{R}^{D}$ with
scale $\gamma \in \mathbb{R}^{D}$ and shift $\beta \in \mathbb{R}^{D}$:

$$\mu = \frac{1}{D}\sum_{j=1}^{D} x_j, \qquad
  \sigma^2 = \frac{1}{D}\sum_{j=1}^{D} (x_j - \mu)^2 ,$$

$$y_j = \gamma_j \cdot \frac{x_j - \mu}{\sqrt{\sigma^2 + \varepsilon}} + \beta_j .$$

The variance is the **biased / population** estimator (divide by $D$, not
$D-1$) — this is what PyTorch and every transformer implementation use.

A naive implementation loops over the $N$ rows (and often the $D$ features) in
Python, normalizing one row and applying $\gamma, \beta$ per element. That is
$O(ND)$ interpreted operations, and it also computes the normalization and the
affine in two separate passes. A single vectorized expression fuses both steps:
compute $\mu$ and $\sigma^2$ per row with a reduction over the last axis, then
apply the standardization and the affine in one broadcasted pass — the Python
interpreter issues only a handful of statements while NumPy does all the work in
C.

## Task

Implement `layernorm(x, gamma, beta, eps=1e-5)`:

```python
def layernorm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
              eps: float = 1e-5) -> np.ndarray:
    ...
```

- `x` has shape $(N, D)$; `gamma` and `beta` have shape $(D,)$.
- Normalize each row over the **last axis** using the biased variance, then
  apply the affine transform $\gamma, \beta$.
- Fuse the normalize and affine steps: use vectorized NumPy only, with **no
  Python `for`/`while` loop** over rows or features.
- Return a `float64` array of shape $(N, D)$.

## Example

```python
import numpy as np
x = np.array([[1.0, 2.0, 3.0]])
gamma = np.array([1.0, 1.0, 1.0])
beta = np.array([0.0, 0.0, 0.0])
y = layernorm(x, gamma, beta, eps=1e-5)
# row mean = 2, biased var = 2/3
# y ≈ [[-1.2247, 0.0, 1.2247]]
```

## What the gate checks

Two gates:

- $\mathrm{max\_abs\_err}$ — the maximum absolute difference against a NumPy
  reference LayerNorm must satisfy $\mathrm{max\_abs\_err} \le 10^{-6}$. This
  requires float64 math; a float32 pass will not be accurate enough.
- $\mathrm{op\_count}$ — the number of Python line events recorded by a tracer
  while `layernorm` runs must satisfy $\mathrm{op\_count} \le 50$. A vectorized
  solution emits only a handful of line events; any Python loop over rows or
  features emits far more than $50$ and fails.
