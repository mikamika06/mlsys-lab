## Context

Activation-aware weight quantization (AWQ) searches for a rescaling factor that
reduces the output error after quantization. Given a weight matrix
$W \in \mathbb{R}^{m \times n}$ and calibration activations
$X \in \mathbb{R}^{n \times k}$, a per-input-channel scale vector
$s \in \mathbb{R}^{n}$ produces the quantized approximation

$$
\hat{Y} = Q(W \operatorname{diag}(s)) \operatorname{diag}(s)^{-1} X .
$$

The objective is to minimize the output reconstruction error

$$
\operatorname{MSE}(s) =
\frac{1}{mk}\lVert WX - Q(W\operatorname{diag}(s))
\operatorname{diag}(s)^{-1}X\rVert_F^2 .
$$

This task uses a grid search over scale exponents. For each exponent
$\alpha$, the scale is

$$
s_i = (\max_j |W_{ji}| + \epsilon)^\alpha .
$$

The quantizer uses symmetric int8 rounding:

$$
Q(A) = \operatorname{clip}(\operatorname{round}(A / z), -127, 127)z,
$$

where $z = \max(|A|)/127$.

The best exponent is the one with the lowest output MSE.

## Task

Implement `awq_grid_scale(W, X, steps=41)`:

```python
def awq_grid_scale(W: np.ndarray, X: np.ndarray, steps: int = 41) -> np.ndarray:
    ...
```

Return the scale vector $s$ with shape `(W.shape[1],)` that gives the lowest
reconstruction MSE among exponents uniformly sampled from $0$ to $1$ using
`steps` points.

Use NumPy operations. The returned values must be `float64`.

## Example

```python
import numpy as np

W = np.array([[1.0, 2.0], [3.0, 1.0]])
X = np.array([[1.0], [0.5]])

s = awq_grid_scale(W, X, steps=11)
Y_hat = quantize_scaled_output(W, X, s)
```

The selected scale is the one whose quantized output is closest to the original
matrix product $WX$.

## What the gate checks

The grader builds its own calibration matrices and computes the oracle answer by
performing the same AWQ grid search and NumPy quantization procedure directly.
The returned scale is evaluated by the resulting output MSE gap from the oracle
minimum. A correct implementation has a gap of at most
$10^{-12}$.
