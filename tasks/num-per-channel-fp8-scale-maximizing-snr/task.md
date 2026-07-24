## Context

Low-precision inference often stores neural network weights using formats such as fp16, bf16, and fp8. FP8 E4M3 represents values with a sign bit, four exponent bits, and three mantissa bits. A common approach is to use a separate scale for each output channel so that each row of a weight matrix uses the available FP8 range effectively.

For a channel $w \in \mathbb{R}^n$, quantization with scale $s$ is

$$
q = Q_{\mathrm{E4M3}}\left(\frac{w}{s}\right), \qquad \hat{w} = s q .
$$

The scale affects the rounding error. The SNR-maximizing scale is the one that minimizes reconstruction error:

$$
s^* = \arg\min_s \lVert w - sQ_{\mathrm{E4M3}}(w/s)\rVert^2 .
$$

For a matrix $W \in \mathbb{R}^{m \times n}$, each row is treated as an independent channel and receives its own optimized scale.

## Task

Implement `fp8_channel_quantize(W)`:

```python
def fp8_channel_quantize(W: np.ndarray) -> np.ndarray:
    ...
```

The input is a 2-D floating point NumPy array. Return a float64 array with the same shape containing the dequantized values after per-channel FP8 E4M3 quantization.

Use this deterministic E4M3 representation:
- Positive finite values are
  $$2^e(1+m/8)$$
  where $e \in \{-6,\dots,7\}$ and $m \in \{0,\dots,7\}$.
- Zero is representable.
- Quantization selects the nearest representable magnitude and preserves the sign.
- Values above the maximum finite value clamp to that maximum.

For each row, choose the scale that minimizes the squared reconstruction error. A deterministic search over candidate scales is acceptable.

## Example

```python
import numpy as np

W = np.array([
    [1.0, 2.0, 8.0],
    [0.1, 0.3, 0.9],
])

W_hat = fp8_channel_quantize(W)
```

`W_hat` has the same shape as `W`, with each row quantized using its own optimized FP8 scale.

## What the gate checks

The gate independently computes an FP8 E4M3 oracle using the same mathematical optimization procedure and compares the submitted reconstruction against that oracle.

The reported metric is

$$
\max_i \frac{\lVert \hat{w}_i-\hat{w}^{\,\mathrm{oracle}}_i\rVert_2}
{\lVert \hat{w}^{\,\mathrm{oracle}}_i\rVert_2+10^{-12}} .
$$

The value must be at most $10^{-12}$. Using a single scale for the whole tensor fails because channels with different magnitudes require different quantization ranges.
