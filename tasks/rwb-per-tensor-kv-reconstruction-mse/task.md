## Context

FP8 KV-cache quantization (e.g. vLLM's `kv_cache_dtype=fp8`) casts K and
V to the E4M3 format (4 exponent bits, 3 mantissa bits, max representable
magnitude 448) using a **per-tensor absmax scale**:

$$
s = \frac{\max_{i}|x_i|}{448}, \qquad
\hat{x}_i = \text{E4M3}\!\left(\frac{x_i}{s}\right) \cdot s
$$

where $\text{E4M3}(\cdot)$ rounds to the nearest representable E4M3
value (clipped to $\pm 448$). Because K and V generally have different
magnitude distributions, each tensor gets **its own independent** scale
— reusing K's scale for V (or vice versa) wastes mantissa precision on
whichever tensor is smaller.

The reconstruction error introduced by this cast is measured with mean
squared error:

$$
\text{MSE}(X) = \frac{1}{|X|}\sum_i (\hat{x}_i - x_i)^2
$$

## Task

Implement `kv_fp8_reconstruction_mse`:

```python
def kv_fp8_reconstruction_mse(K: np.ndarray, V: np.ndarray) -> dict:
    ...
```

- `K`, `V`: arbitrary-shape float arrays (independent tensors).
- Compute a separate per-tensor absmax scale for `K` and for `V`,
  quantize each to E4M3 with its own scale, dequantize, and compute each
  tensor's reconstruction MSE.
- Return `{"mse_k": float, "mse_v": float}`.

## Example

```python
import numpy as np

K = np.random.default_rng(0).standard_normal((64, 16)) * 2.0
V = np.random.default_rng(1).standard_normal((64, 16)) * 0.3

out = kv_fp8_reconstruction_mse(K, V)
# out["mse_k"], out["mse_v"] are small positive floats; because V has a
# smaller dynamic range than K, its scale is smaller and its relative
# quantization error is generally comparable, not identical, to K's.
```

## What the gate checks

The grader loads a committed `k.npy`/`v.npy` fixture (K/V-shaped tensors
with a few high-magnitude outlier channels, similar to real transformer
activation statistics) plus several additional seeded synthetic tensor
pairs, and computes each tensor's reference MSE with an independent
NumPy implementation of the same absmax-scaled E4M3 round-trip — never
calling your function, never hardcoding an expected value.

`rel_err` is `scorers.rel_err` applied to the 2-vector
`[mse_k, mse_v]` against the oracle's, taking the worst case across all
tensor pairs, and must be `<= 1e-6`. Sharing one scale between K and V,
using the wrong exponent/mantissa clipping, or omitting the denormal
(sub-`2^-6`) branch of the E4M3 grid will all shift the reconstruction
error measurably.
