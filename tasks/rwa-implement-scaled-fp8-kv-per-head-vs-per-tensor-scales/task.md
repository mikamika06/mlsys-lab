## Context

FP8 inference stores tensors using 8-bit floating point values and a scale factor. A common format is e4m3, which has 4 exponent bits and 3 mantissa bits with a maximum finite magnitude of $448$.

For a tensor $X$, scaled FP8 quantization uses

$$s = \frac{\max(|X|)}{448},$$

then stores

$$Q = \operatorname{fp8e4m3}\left(\frac{X}{s}\right).$$

The approximate reconstruction is

$$\hat{X} = sQ.$$

Attention computes

$$\operatorname{Attn}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V.$$

For KV caches in transformer inference, different attention heads can have different value ranges. A single tensor-wide scale is affected by outlier heads because the largest value determines the scale for all heads. Per-head scales instead compute

$$s_h = \frac{\max(|X_h|)}{448}$$

independently for each head $h$.

## Task

Implement `scaled_fp8_kv_attention(K, V, Q, per_head)`:

```python
def scaled_fp8_kv_attention(K: list[list[list[float]]], V: list[list[list[float]]], Q: list[list[list[float]]], per_head: bool) -> list[list[list[float]]]:
    ...
```

The inputs have shapes $(H, N, D)$ for `K` and `V`, and $(H, M, D)$ for `Q`.

The function must:

1. Quantize `K` and `V` to simulated e4m3 FP8 values using a scale of `amax / 448`.
2. Use one scale for the whole tensor when `per_head` is `False`.
3. Use separate scales for each head when `per_head` is `True`.
4. Dequantize `K` and `V`.
5. Compute attention using the dequantized tensors in float64.
6. Return an array of shape $(H, M, D)$.

The implementation should use Python operations and may use helper functions inside the file.

## Example

```python

K = [[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]
V = [[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]]
Q = [[[1.0, 1.0]], [[1.0, 1.0]]]

out = scaled_fp8_kv_attention(K, V, Q, True)
# shape is (2, 1, 2)
```

## What the gate checks

The gate builds a Python oracle that simulates e4m3 quantization, dequantization, and attention in float64. The returned attention output must have maximum absolute error $\le 10^{-3}$ against the oracle.

A second check uses KV tensors with a large outlier range difference between heads. The per-head implementation must produce a strictly lower reconstruction error than a per-tensor scale on the outlier-heavy case. A solution that always uses one global scale fails this check.
