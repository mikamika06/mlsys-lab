## Context

Casting a weight matrix to fp8 E4M3 (max finite magnitude $448$) needs a
scale per value it's applied to, so the tensor's magnitudes land near
the top of E4M3's grid instead of being crushed toward zero (or clipped
at the top) by a mismatched scale. A single scale for the whole weight
matrix is wasted on any row whose magnitude range differs from the
row that has the single largest value. **Per-channel** (per-output-row)
scaling gives each output row its own scale, so a quiet row and a loud
row each get the full resolution E4M3 can offer:

$$
\text{scale}_i = \frac{\max_j |W_{i,j}|}{448}, \qquad
\hat{W}_{i,j} = \mathrm{round\_to\_e4m3}\!\left(\frac{W_{i,j}}{\text{scale}_i}\right) \cdot \text{scale}_i
$$

where $\mathrm{round\_to\_e4m3}$ rounds a value's magnitude to the
**nearest** representable E4M3 grid point (sign preserved, magnitude
clipped to $448$). A row that is entirely zero has an undefined scale by
this formula — treat it as $\text{scale}_i = 1.0$ (its values are all
zero regardless of scale).

## Task

Implement `per_channel_fp8_quant`:

```python
def per_channel_fp8_quant(W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ...
```

- `W`: a `(rows, cols)` float64 weight matrix.

Return `(scales, W_dequant)`:

- `scales`: `(rows,)` — each row's scale, per the formula above (`1.0`
  for an all-zero row).
- `W_dequant`: `(rows, cols)` — the dequantized reconstruction of `W`
  after per-row E4M3 quantize/dequantize, per the formula above.

## Example

```python
import numpy as np

W = np.array([[100.0, -100.0, 50.0], [0.01, -0.02, 0.0]])
scales, W_dequant = per_channel_fp8_quant(W)
# row 0's scale = 100/448 (small, since its own magnitudes are all large)
# row 1's scale = 0.02/448 (tiny, sized for its own much smaller values)
# a single shared scale (sized for row 0) would have crushed row 1's
# already-small values even further toward zero.
```

## What the gate checks

The grader builds several `(rows, cols)` weight matrices from a seeded
NumPy generator — rows with wildly different magnitude ranges, a row
that is exactly all zeros, and a mix of positive/negative values — and
computes the reference `(scales, W_dequant)` independently in NumPy: it
builds the *real* E4M3 grid from the sign/exponent/mantissa bit-layout
formulas (decoding every representable code — the same oracle a hardware
cast would produce), then reproduces the per-row absmax-scale-and-round
procedure above, never calling your function.

`max_abs_err` is the worst-case elementwise absolute error across both
your `scales` and your `W_dequant`, over every scenario, and the gate
requires `<= 1e-4`. Sharing one scale across every row instead of
per-row, dividing by `cols` or by the matrix-wide max instead of the
row's own max, rounding down/up instead of to the nearest grid point, or
mishandling the all-zero row (dividing by zero) will all produce a
visible mismatch.
