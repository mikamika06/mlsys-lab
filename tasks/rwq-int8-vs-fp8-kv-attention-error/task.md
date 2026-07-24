## Context

Transformer attention computes a weighted combination of value vectors. For a query matrix $Q$, key matrix $K$, and value matrix $V$, scaled dot-product attention is

$$
\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V .
$$

Production inference systems often reduce KV-cache memory by storing keys and values in lower precision formats. This task compares two simulated formats: signed INT8 quantization and an FP8-like $e4m3$ quantization.

For a tensor $X$, symmetric INT8 quantization uses

$$
s=\frac{\max(|X|)}{127}, \qquad X_q=\mathrm{round}\left(\frac{X}{s}\right),
$$

and reconstructs

$$
\hat{X}=sX_q .
$$

The attention error is measured against full precision attention output using mean squared error:

$$
\mathrm{MSE}(A,B)=\frac{1}{N}\sum_i(A_i-B_i)^2 .
$$

The FP8 simulation uses a deterministic $e4m3$ style representation by rounding values to the nearest representable power-scaled mantissa values.

## Task

Implement `kv_attention_quant_error(Q, K, V)`.

```python
def kv_attention_quant_error(Q: np.ndarray, K: np.ndarray, V: np.ndarray):
    ...
```

The inputs are `float32` arrays with shape $(n,d)$ for $Q$ and $K$, and shape $(n,d_v)$ for $V$.

Return a tuple:

```python
(int8_mse, fp8_mse, winner)
```

where:

- `int8_mse` is the MSE between full precision attention output and attention using INT8-quantized $K,V$.
- `fp8_mse` is the MSE between full precision attention output and attention using FP8-quantized $K,V$.
- `winner` is the string `"int8"` if INT8 has lower or equal error, otherwise `"fp8"`.

Use NumPy operations only.

## Example

```python
import numpy as np

Q = np.array([[0.2, -0.1], [0.4, 0.3]], dtype=np.float32)
K = np.array([[0.1, 0.5], [-0.2, 0.4]], dtype=np.float32)
V = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

int8_err, fp8_err, winner = kv_attention_quant_error(Q, K, V)
```

## What the gate checks

The gate recomputes the full precision, INT8, and FP8 attention outputs with a NumPy oracle. The returned MSE values must match the oracle within numerical tolerance, and the reported winner must match the oracle winner.

## Grading helpers available (arena/scorers.py):

```python
from arena import scorers
```
