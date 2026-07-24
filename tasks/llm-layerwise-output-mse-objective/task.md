## Context

Post-training quantization replaces a full precision weight matrix $W$ with a
lower precision approximation $W_q$. The goal is to minimize the change in the
output of the layer on representative calibration activations.

For a linear layer with activations $X$, the original output is

$$Y = WX$$

and the quantized output is

$$Y_q = W_qX.$$

The layerwise output error objective is the mean squared difference between these
outputs:

$$
\mathrm{MSE}(W, W_q, X) =
\frac{1}{mn}\sum_{i=1}^{m}\sum_{j=1}^{n}(Y_{ij}-Y_{q,ij})^2 .
$$

This measures output-space error rather than only weight-space error. A small
weight change can create a large output error when activations have large
magnitudes.

## Task

Implement `layerwise_output_mse(W, W_q, X)`:

```python
def layerwise_output_mse(W: np.ndarray, W_q: np.ndarray, X: np.ndarray) -> float:
    ...
```

The inputs are NumPy arrays:

- `W` is the original weight matrix with shape $(m, k)$.
- `W_q` is the quantized weight matrix with the same shape as $W$.
- `X` is the activation matrix with shape $(k, n)$.

Return the Python `float` value of the mean squared error between `W @ X` and
`W_q @ X`.

## Example

```python
import numpy as np

W = np.array([[1.0, 2.0], [3.0, 4.0]])
W_q = np.array([[1.0, 1.5], [3.0, 3.5]])
X = np.array([[1.0, 2.0], [0.5, 1.0]])

value = layerwise_output_mse(W, W_q, X)
```

The function compares the layer outputs before and after quantization and
returns their average squared difference.

## What the gate checks

The gate computes the expected objective using NumPy matrix multiplication and
mean squared error as the numerical oracle. The returned value is compared with
that oracle. The final metric is the squared difference between the submitted
result and the oracle result, which must satisfy

$$\mathrm{MSE}_{gate} \le 10^{-12}.$$
