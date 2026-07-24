## Context

Attention uses queries $Q$, keys $K$, and values $V$. For a single attention head, the
output is

$$
Y = \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V .
$$

Production inference systems often reduce KV cache memory by storing keys and values
in FP8. A common approach is per-token scaling: each token vector $x$ is scaled by
its own maximum magnitude.

For a token vector $x$, the scale is

$$
s = \frac{\max_i |x_i|}{448},
$$

where $448$ is the largest finite value in the E4M3 FP8 format. The quantized value is

$$
q = \operatorname{FP8}_{\mathrm{E4M3}}\left(\frac{x}{s}\right),
$$

and dequantization reconstructs

$$
\hat{x} = q s .
$$

The attention computation then uses dequantized keys and values:

$$
\hat{Y} =
\operatorname{softmax}\left(\frac{Q\hat{K}^{\top}}{\sqrt{d}}\right)\hat{V}.
$$

The quality error is measured against the original FP32 attention output.

## Task

Implement `fp8_kv_attention(Q, K, V)`.

```python
def fp8_kv_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray):
    ...
```

Inputs are 2-D NumPy arrays with shapes $(m,d)$, $(n,d)$, and $(n,d_v)$.
The function must:

1. Compute per-token E4M3 quantization for rows of $K$ and $V$.
2. Dequantize the FP8 values back to float64.
3. Compute attention using the dequantized KV tensors.
4. Return a tuple `(Y, mse)` where `Y` is the attention output and `mse` is the
   mean squared error between `Y` and the full precision attention output.

Use NumPy operations only.

## Example

```python
import numpy as np

Q = np.array([[1.0, 0.0]])
K = np.array([[1.0, 0.0], [0.0, 1.0]])
V = np.array([[2.0], [4.0]])

Y, mse = fp8_kv_attention(Q, K, V)
```

The returned `Y` contains the FP8-KV attention result and `mse` is the numerical
difference from computing attention with the original $K$ and $V$.

## What the gate checks

The gate builds a NumPy oracle that performs the per-token E4M3 quantization,
dequantization, and attention calculation. The returned output must have relative
error

$$
\frac{\lVert Y-\hat{Y}\rVert}{\lVert \hat{Y}\rVert+10^{-12}}
\le 10^{-4}.
$$

The reported MSE must match the oracle-computed MSE within the gate tolerance.
Using a different scaling rule, FP16 instead of E4M3, or global rather than per-token
scales will fail.
