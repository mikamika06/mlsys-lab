## Context

KV-cache quantization has to pick, for each tensor, which axis shares a
scale/zero-point. KIVI (2-bit KV cache quantization) shows the two
tensors need **different** axis choices:

- **Keys** are quantized **per-channel**: one scale/zero-point per
  column $j$, fit across all tokens. This works because RoPE-rotated key
  channels have consistent, channel-specific outlier magnitude — some
  columns of $K$ are just "loud" across every token, so giving each
  column its own range captures that structure.
- **Values** are quantized **per-token**: one scale/zero-point per row
  $i$, fit across all channels. Value outliers are token-specific, not
  channel-specific, so per-row grouping is the axis that matches the
  outlier structure there.

For a group of values $\{x_k\}$ quantized to $b$ bits with uniform
affine (asymmetric) quantization, the min-max quantizer is

$$
s = \frac{\max_k x_k - \min_k x_k}{2^{b}-1}, \qquad
z = \operatorname{round}\!\left(\frac{-\min_k x_k}{s}\right), \qquad
\hat x_k = s\left(\operatorname{clip}\!\left(\operatorname{round}\!\left(\frac{x_k}{s}\right) + z,\ 0,\ 2^{b}-1\right) - z\right).
$$

Because a per-channel group's value range is always a *subset* of the
full tensor's range, giving each channel its own $s, z$ can only match
or shrink the quantization step compared to sharing one $s, z$ across
the whole tensor — so per-channel key quantization never has *higher*
reconstruction error than per-tensor key quantization.

## Task

Implement `kivi_quant_errors`:

```python
def kivi_quant_errors(K: np.ndarray, V: np.ndarray, q: np.ndarray, bits: int) -> np.ndarray:
    ...
```

* `K`, `V` — shape $(n, d)$, fp64 key/value cache.
* `q` — shape $(d,)$, fp64 query vector.
* `bits` — quantizer bit-width.

Using the affine min-max quantizer above:

1. Quantize+dequantize `K` **per-channel** (one scale/zero-point per
   column, min/max taken over all $n$ rows).
2. Quantize+dequantize `K` **per-tensor** (one global scale/zero-point)
   as a baseline for comparison.
3. Quantize+dequantize `V` **per-token** (one scale/zero-point per row,
   min/max taken over all $d$ columns).
4. Compute standard softmax attention,
   $\operatorname{softmax}(Kq/\sqrt d)^\top V$, once with the exact
   fp64 `K, V` and once with the (per-channel `K`, per-token `V`)
   dequantized cache.

Return `np.array([k_mse_per_channel, k_mse_per_tensor, attn_max_abs_err])` where:

* `k_mse_per_channel` — MSE between the per-channel-dequantized `K` and
  the original `K`.
* `k_mse_per_tensor` — MSE between the per-tensor-dequantized `K` and
  the original `K`.
* `attn_max_abs_err` — max absolute error between the KIVI-quantized
  attention output and the exact fp64 attention output.

## Example

```python
import numpy as np

rng = np.random.default_rng(0)
K = rng.standard_normal((16, 8))
V = rng.standard_normal((16, 8))
q = rng.standard_normal(8)

out = kivi_quant_errors(K, V, q, bits=4)
# out[0] (per-channel K MSE) < out[1] (per-tensor K MSE)
```

## What the gate checks

The gate, **max_abs_err**, compares your 3-element array against an
fp64 NumPy oracle across several `(n, d, bits)` configurations,
including low bit-widths down to 2 bits. The oracle always finds
`k_mse_per_channel < k_mse_per_tensor` for these cases — mixing up the
per-channel/per-token axis assignment (e.g. quantizing keys per-token or
values per-channel) or falling back to a single per-tensor scale for
everything produces a measurably different `k_mse_per_channel` and
`attn_max_abs_err`, which fails the gate.
