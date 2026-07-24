## Context

In second-order quantization methods such as GPTQ, weight updates are
guided by the Hessian of the layer's output loss.  For a linear layer
whose input activations are the rows of a matrix $X \in \mathbb{R}^{n \times d}$,
the approximate (Fisher) Hessian is

$$H = 2\, X^\top X \;\in\; \mathbb{R}^{d \times d}.$$

The factor of 2 comes from the quadratic Taylor expansion of a
squared-error loss.

Direct inversion of $H$ is numerically unstable when it is
near-singular.  A standard fix is **damping**: add a scalar multiple of
the identity matrix.  The damping scalar is chosen proportional to the
average diagonal element:

$$\text{damp} = \alpha \cdot \frac{1}{d} \sum_{i=1}^{d} H_{ii},$$

where $\alpha$ (called `percent` here) is typically a small fraction such
as $0.01$.  The damped Hessian used in practice is then
$H_{\text{damped}} = H + \text{damp} \cdot I$.

## Task

Implement `damped_hessian(X, percent=0.01)`:

```python
def damped_hessian(X, percent=0.01):
    """Compute approximate Hessian and diagonal damping scalar.

    Args:
        X: numpy array of shape (n, d) — calibration input activations.
        percent: float — damping fraction (default 0.01).

    Returns:
        H: numpy array of shape (d, d) — the approximate Hessian 2 * X^T @ X.
        damp: float — the damping scalar percent * mean(diag(H)).
    """
```

Use NumPy only.  The function must return a tuple `(H, damp)` where `H`
is a `float64` array of shape `(d, d)` and `damp` is a Python float.

## Example

```python
import numpy as np
X = np.array([[1.0, 0.0],
              [0.0, 1.0],
              [1.0, 1.0]])
H, damp = damped_hessian(X, percent=0.01)
# H = 2 * [[2, 1], [1, 2]] = [[4, 2], [2, 4]]
# mean(diag(H)) = mean([4, 4]) = 4.0
# damp = 0.01 * 4.0 = 0.04
```

## What the gate checks

The grader generates a $(200, 64)$ matrix of calibration activations
using a fixed random seed, then recomputes $H = 2\, X^\top X$ and
$\text{damp} = 0.01 \cdot \text{mean}(\text{diag}(H))$ with a NumPy
oracle.  It reports two relative errors:

$$\text{rel\_err\_h} = \frac{\lVert H_{\text{learner}} - H_{\text{ref}} \rVert_F}{\lVert H_{\text{ref}} \rVert_F + \varepsilon}, \qquad \text{rel\_err\_damp} = \frac{|\text{damp}_{\text{learner}} - \text{damp}_{\text{ref}}|}{|\text{damp}_{\text{ref}}| + \varepsilon}.$$

Both must be $\le 10^{-8}$ to pass.
