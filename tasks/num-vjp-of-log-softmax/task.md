## Context

`log_softmax` maps each row of a matrix $x \in \mathbb{R}^{B \times C}$ to

$$
y_i = x_i - \log\sum_{k=1}^{C} e^{x_k}, \qquad i = 1,\dots,C
$$

(applied independently per row; the max-shift used for numerical stability
doesn't change the value and is omitted here for clarity).

In reverse-mode autodiff, a **vector-Jacobian product (VJP)** takes the
upstream gradient $g = \partial \mathcal{L}/\partial y$ (same shape as $y$)
and produces $\partial \mathcal{L}/\partial x = g^\top J$, where $J$ is the
Jacobian of $y$ with respect to $x$. For log-softmax this has a compact
closed form. Writing $p = \mathrm{softmax}(x) = e^{y}$:

$$
\frac{\partial \mathcal{L}}{\partial x_i} = g_i - p_i \sum_{k=1}^{C} g_k .
$$

Intuitively: the gradient passes straight through ($g_i$), minus a
correction proportional to the softmax weight $p_i$ and the *total* upstream
gradient mass $\sum_k g_k$ — because every output row depends on the whole
input row through the shared normalizer.

## Task

Implement `log_softmax_vjp`:

```python
def log_softmax_vjp(x, g):
    """Vector-Jacobian product of y = log_softmax(x, axis=-1).

    Given the upstream gradient `g` (dLoss/dy, same shape as x), returns
    dLoss/dx, a float64 array the same shape as x.
    """
```

* `x` — a 2-D NumPy array of shape $(B, C)$ (log-softmax is applied per row,
  i.e. along the last axis).
* `g` — a NumPy array of the same shape as `x`: the upstream gradient.
* Return `g - softmax(x) * sum(g, axis=-1, keepdims=True)`, computing
  `softmax(x)` yourself (e.g. via a numerically-stable log-softmax).

## Example

```python
import numpy as np
x = np.array([[1.0, 2.0, 3.0]])
g = np.array([[1.0, 0.0, 0.0]])
log_softmax_vjp(x, g)
# softmax(x) ≈ [[0.0900, 0.2447, 0.6652]]
# result ≈ [[1 - 0.0900, 0 - 0.2447, 0 - 0.6652]]
#         = [[0.9100, -0.2447, -0.6652]]
```

## What the gate checks

The grader never uses the closed-form formula above as its oracle. Instead
it defines the scalar $s(x) = \sum_{i,k} g_{ik}\, \big(\log\text{-softmax}(x)\big)_{ik}$
and estimates $\partial s/\partial x_{ik}$ independently with **central
finite differences**:

$$
\left(\frac{\partial s}{\partial x}\right)_{ik} \approx \frac{s(x + h\,e_{ik}) - s(x - h\,e_{ik})}{2h}, \qquad h = 10^{-5}.
$$

This numerical gradient is mathematically identical to the VJP you're
computing (that's what makes VJPs checkable this way), but it never touches
your softmax code. **max_abs_err** is the maximum absolute difference
between your `log_softmax_vjp` output and this finite-difference estimate,
taken over four random `(x, g)` pairs of different shapes, and must satisfy
$\mathrm{max\_abs\_err} \le 10^{-5}$.
