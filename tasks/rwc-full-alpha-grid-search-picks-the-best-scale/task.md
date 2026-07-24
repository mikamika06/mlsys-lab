## Context

Activation-aware Weight Quantization (AWQ) searches for a scaling factor that reduces the
error introduced by quantization. For a weight matrix $W$ and calibration activations
$X$, a scale vector is applied before quantization:

$$
W' = W \operatorname{diag}(s).
$$

After quantizing the scaled weights, the original output is approximated by undoing the
scale:

$$
Y_s = Q(W \operatorname{diag}(s)) \operatorname{diag}(1/s)X .
$$

The scale is derived from activation statistics. A common AWQ parameterization uses

$$
s = s_X^\alpha ,
$$

where $s_X$ contains activation magnitudes and $\alpha$ is selected from a fixed grid.
The best $\alpha$ minimizes the output reconstruction error:

$$
\operatorname{loss}(\alpha) =
\left\lVert WX -
Q(W \operatorname{diag}(s_X^\alpha))
\operatorname{diag}(s_X^{-\alpha})X
\right\rVert_2 .
$$

A production implementation evaluates the full alpha grid and selects the index with the
lowest loss rather than assuming a fixed alpha.

## Task

Implement `search_awq_alpha(W, X, s_x)`.

```python
def search_awq_alpha(
    W: np.ndarray,
    X: np.ndarray,
    s_x: np.ndarray,
) -> tuple[int, np.ndarray]:
    ...
```

The function receives a weight matrix $W \in \mathbb{R}^{m \times n}$, calibration
activations $X \in \mathbb{R}^{n \times k}$, and activation statistics
$s_X \in \mathbb{R}^{n}$.

Use the 20 point alpha grid:

$$
[0.0, 0.05, 0.10, \dots, 0.95].
$$

For every alpha value, compute $s=s_X^\alpha$, quantize the scaled weights using
symmetric per-row int8 quantization, reconstruct the output, and measure the Frobenius
norm error. Return the index of the alpha with minimum error and the complete loss curve
as a NumPy array of length $20$.

The quantization rule is:

$$
q_i = \operatorname{round}
\left(
\frac{w_i}{\max(|w_i|)/127}
\right),
\qquad
\hat{w}_i =
q_i \frac{\max(|w_i|)}{127},
$$

where each row is quantized independently.

Assume all inputs are finite and $s_X$ values are positive.

## Example

```python
import numpy as np

W = np.array([[1.0, -2.0], [0.5, 3.0]])
X = np.array([[1.0], [2.0]])
s_x = np.array([1.5, 0.8])

idx, losses = search_awq_alpha(W, X, s_x)

# idx is the best alpha grid position
# losses contains 20 reconstruction errors
```

## What the gate checks

The gate builds an independent NumPy oracle that evaluates the same 20 alpha candidates
and symmetric int8 quantization procedure. It checks that the returned index matches the
oracle argmin and that the complete loss curve has negligible relative error.

A shortcut that always returns a fixed alpha, skips the grid search, or uses an incorrect
quantization reconstruction will fail.
