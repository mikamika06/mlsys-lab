## Context

Production attention systems often compress the key-value (KV) cache to reduce memory use. A common approach is symmetric int8 quantization. For a vector $x$, the scale is computed from the maximum absolute value:

$$s = \frac{\max(|x|)}{127}.$$

The quantized values are

$$q = \mathrm{round}\left(\frac{x}{s}\right),$$

and reconstruction uses

$$\hat{x} = q\,s.$$

When a KV cache contains multiple tokens, each token has its own key and value vectors. A single scale over the entire cache can waste precision because large-magnitude tokens dominate the range. Per-token quantization instead computes one scale for each token row.

Given key cache $K \in \mathbb{R}^{n \times d}$, value cache $V \in \mathbb{R}^{n \times d}$, and query $Q \in \mathbb{R}^{m \times d}$, attention is

$$
A = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right),
$$

$$
O = AV.
$$

The quantized path should quantize each row of $K$ and $V$ independently, then run attention on the reconstructed arrays.

## Task

Implement `quantized_kv_attention(Q, K, V)`.

```python
def quantized_kv_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray
) -> np.ndarray:
    ...
```

The function receives float arrays with shapes $(m,d)$, $(n,d)$, and $(n,d)$. It must:

1. Quantize every row of `K` and `V` separately using symmetric int8 quantization.
2. Dequantize the rows back to float64.
3. Compute scaled dot-product attention using the dequantized cache.
4. Return the attention output as a float64 NumPy array of shape $(m,d)$.

Do not use one global scale for the whole cache. Handle rows whose maximum absolute value is zero.

## Example

```python
import numpy as np

Q = np.array([[0.5, -0.2]])
K = np.array([[1.0, 0.0], [-0.5, 2.0]])
V = np.array([[1.0, 2.0], [3.0, 4.0]])

out = quantized_kv_attention(Q, K, V)
```

The result is the attention output produced by quantizing and reconstructing the two cache matrices before applying attention.

## What the gate checks

The gate computes a NumPy oracle implementation of per-token KV quantization and attention. The returned result must have relative error

$$
\frac{\lVert y-\hat{y}\rVert}{\lVert y\rVert}
$$

at most $10^{-5}$ compared with the oracle output.

The gate also computes the mean squared error of a buggy implementation that uses one scale for the entire cache. The submitted implementation must improve on that baseline, so its MSE divided by the buggy MSE must be less than $1$.
