## Context

Symmetric int8 quantization maps a real-valued tensor $w$ to 8-bit
integer codes using a single per-tensor scale:

$$
s = \frac{\max_i |w_i|}{127}, \qquad
c_i = \mathrm{clip}\!\left(\mathrm{round}\!\left(\frac{w_i}{s}\right),\, -127,\, 127\right) .
$$

Dequantization is the inverse map:

$$
\hat w_i = c_i \cdot s .
$$

Because $s$ is derived from the tensor's own maximum magnitude, no code
ever needs to clip (the largest-magnitude element maps to exactly
$\pm 127$), and round-to-nearest guarantees a tight, provable error bound
for every element:

$$
|\hat w_i - w_i| \le \frac{s}{2} .
$$

This is the simplest possible weight/KV-cache quantization scheme, and
the one every fancier scheme (per-channel, per-token, asymmetric, etc.)
builds on.

## Task

Implement both directions of the codec:

```python
def quantize_symmetric_int8(w: np.ndarray):
    ...
    # returns (codes, scale): codes is an int8 array shaped like `w`,
    # scale is a Python float

def dequantize_symmetric_int8(codes: np.ndarray, scale: float) -> np.ndarray:
    ...
    # returns a float32 array shaped like `codes`
```

* `quantize_symmetric_int8(w)` computes `scale = max(abs(w)) / 127` (use
  `scale = 1.0` when `w` is all zeros, purely to avoid a division by
  zero — every code will be `0` regardless), then rounds `w / scale` to
  the nearest integer, clips to `[-127, 127]`, and casts to `int8`.
* `dequantize_symmetric_int8(codes, scale)` returns
  `codes.astype(float32) * scale`.

## Example

```python
import numpy as np
w = np.array([-4.0, 0.0, 1.5, 4.0], dtype=np.float32)

codes, scale = quantize_symmetric_int8(w)
print(codes, scale)          # [-127    0   48  127] 0.031496062992125984

w_hat = dequantize_symmetric_int8(codes, scale)
print(w_hat)                  # [-4.        0.        1.511811  4.      ]
```

## What the gate checks

The grader recomputes the reference codes and scale for the fixture
tensor `weights.npy` plus several extra random tensors (different
shapes, dynamic ranges, an all-zero tensor, and a tensor with one huge
outlier) using the exact `max(abs(w)) / 127` formula above — never your
own implementation.

* **`code_exact_fraction`** — the fraction of int8 codes across every
  test tensor that match the reference codec bit-for-bit. Must be `1.0`.
* **`max_abs_err`** — for each tensor, the worst-case round-trip error is
  divided by that tensor's own `scale / 2` bound, and the grader reports
  the maximum of that ratio across all tensors. Must be `<= 1.001` (a
  correct round-to-nearest codec always sits at or just below `1.0`; the
  small slack only absorbs floating-point rounding noise). A codec that
  quantizes without rounding to the nearest code (e.g. truncation), uses
  the wrong scale, or forgets to clip will blow this ratio up by a large
  margin.
