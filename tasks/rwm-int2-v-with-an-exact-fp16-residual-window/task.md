## Context

Long-context inference serving keeps a growing KV cache; compressing the
**value** cache to 2 bits/element is one of the most aggressive real
compression ratios used in production (e.g. KIVI-style KV quantization).
Doing this naively hurts the model badly, because the *most recent* tokens
are disproportionately important for next-token prediction (they are still
being actively attended to and haven't "settled" into the compressed
summary the way older context has).

The standard fix is a **residual window**: keep the last $R$ tokens of the
value cache in full precision (uncompressed, bit-exact), and only quantize
everything *older* than that to int2. As generation proceeds and the window
slides, tokens falling out of the window get quantized once and never
touched again — but the most recent $R$ tokens are always exact.

The quantization used for the non-residual region is **grouped affine
(zero-point) int2 quantization**: for a value-cache row (one token's
$d$-dimensional vector), split it into contiguous groups of `group_size`
channels; each group gets its own affine mapping to the 4 representable
levels $\{0,1,2,3\}$:
$$
\text{lo} = \min(\text{group}),\qquad \text{hi} = \max(\text{group})
$$
$$
\text{scale} = \frac{\text{hi} - \text{lo}}{3}\quad(\text{use scale}=1 \text{ if hi}=\text{lo})
$$
$$
\text{code} = \mathrm{clip}\big(\mathrm{round}((x - \text{lo}) / \text{scale}),\, 0,\, 3\big)
$$
Reconstruction: $\hat{x} = \text{code} \cdot \text{scale} + \text{lo}$.

## Task

Implement:

```python
def kv_int2_residual_window(V: np.ndarray, group_size: int = 32, residual_window: int = 16) -> np.ndarray:
    ...
```

* `V` — 2-D `float64` array of shape `(T, d)`: `T` cached token vectors of
  dimension `d`. `d` is divisible by `group_size`, and `T > residual_window`.
* Split `V` into the first `T - residual_window` rows (the "quantized
  region") and the last `residual_window` rows (the "residual window").
* Quantize the quantized region with the grouped int2 affine scheme above
  (grouping along the channel axis, one `(lo, scale)` pair per group per
  token), then dequantize it back to float.
* Leave the residual window rows **byte-for-byte unchanged**.
* Return the full `(T, d)` reconstructed array: dequantized quantized-region
  rows followed by the untouched residual rows, in original row order.
* Vectorised NumPy only; no explicit Python loops over elements.

## Example

```python
import numpy as np
V = np.random.default_rng(0).standard_normal((96, 64))
V_hat = kv_int2_residual_window(V, group_size=32, residual_window=16)
assert V_hat.shape == V.shape
assert np.array_equal(V_hat[-16:], V[-16:])       # residual window is exact
assert not np.allclose(V_hat[:-16], V[:-16])       # older tokens are lossy
```

## What the gate checks

The grader builds a deterministic `(96, 64)` value-cache matrix (fixed seed)
and calls your function with `group_size=32, residual_window=16`:

* **residual_max_abs_err** — the max absolute difference between your
  output's last 16 rows and the input's last 16 rows must be exactly `0.0`
  (the residual window must never be touched by quantization).
* **quant_max_abs_err** — the max absolute difference between your output's
  first `T - 16` rows and an independent NumPy oracle implementing the exact
  grouped int2 affine scheme above, on the same data, must be effectively
  zero (this is a deterministic table lookup — any correct implementation
  matches the oracle almost exactly).
