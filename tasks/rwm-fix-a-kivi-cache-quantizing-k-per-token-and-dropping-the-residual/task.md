## Context

KIVI-style KV cache quantization reduces memory by storing most cache values in low-bit form while keeping a recent residual window in higher precision.

For this task, a key cache $K \in \mathbb{R}^{B \times H \times T \times D}$ is quantized per channel. Each channel has one scale shared across the token dimension:

$$
s^K_{b,h,d} = \frac{\max_t |K_{b,h,t,d}|}{127}.
$$

The value cache $V \in \mathbb{R}^{B \times H \times T \times D}$ is quantized per token. Each token vector has its own scale:

$$
s^V_{b,h,t} = \frac{\max_d |V_{b,h,t,d}|}{127}.
$$

The int8 representation is

$$
q = \operatorname{round}\left(\frac{x}{s}\right),
$$

and dequantization reconstructs

$$
\hat{x} = q s.
$$

The newest $R$ tokens are not quantized. They are stored as fp16 residuals and restored exactly from that fp16 representation.

## Task

The provided implementation contains a bug in the quantization layout. It quantizes keys with per-token scales and discards the fp16 residual window.

Implement `quantize_dequant_kv_cache(K, V, R)`:

```python
def quantize_dequant_kv_cache(
    K: np.ndarray,
    V: np.ndarray,
    R: int
) -> tuple[np.ndarray, np.ndarray]:
    ...
```

The inputs `K` and `V` have shape $(B,H,T,D)$ and contain floating point values. Return the dequantized key and value caches.

The first $T-R$ tokens must use int8 quantization. Keys must use per-channel scales over $T$, and values must use per-token scales over $D$. The final $R$ tokens must be represented by fp16 residuals and returned after fp16 conversion.

## Example

```python
import numpy as np

K = np.zeros((1, 1, 4, 2), dtype=np.float32)
V = np.ones((1, 1, 4, 2), dtype=np.float32)

K_hat, V_hat = quantize_dequant_kv_cache(K, V, 2)

# The final two tokens are restored from fp16 residuals.
# The earlier tokens are int8 quantized and dequantized.
```

## What the gate checks

The gate builds a NumPy oracle that applies the required KIVI quantization rules. The returned arrays are compared with the oracle reconstruction using the relative error

$$
\mathrm{rel\_err} =
\frac{\lVert \hat{x}_{candidate}-\hat{x}_{oracle}\rVert_2}
{\lVert \hat{x}_{oracle}\rVert_2 + 10^{-12}}.
$$

The residual region is separately checked to ensure the newest $R$ tokens match the fp16 residual representation exactly. A solution that swaps the K and V quantization axes or removes the residual window fails.
