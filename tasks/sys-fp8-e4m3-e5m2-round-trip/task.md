## Context

Modern accelerators (H100-class GPUs and later) support two 8-bit floating
point formats for mixed-precision training and inference, both standardized
by the Open Compute Project:

- **E4M3**: 1 sign bit, 4 exponent bits, 3 mantissa bits. Trades exponent
  range for mantissa precision — used for weights and activations, where
  values are usually within a bounded range but need more significant digits.
  The variant used in practice ("e4m3fn" — *finite*) has no infinities: its
  largest-magnitude finite value is $448$, and values that would overflow
  saturate toward NaN rather than becoming infinite.
- **E5M2**: 1 sign bit, 5 exponent bits, 2 mantissa bits. Trades mantissa
  precision for exponent range — used for gradients, which can span many
  orders of magnitude. Its largest finite value is $57344$.

For a normal (non-subnormal) value with exponent field $e$ (unbiased exponent
$e - \text{bias}$) and $m$-bit mantissa fraction $f \in [0, 1)$:

$$
x = (-1)^s \cdot 2^{\,e - \text{bias}} \cdot (1 + f).
$$

Casting a value into one of these formats (quantization) snaps it to the
nearest representable value in that grid; casting back to `float32`
(dequantization) recovers that snapped value exactly. This round trip is
exactly what happens on real hardware when a tensor is stored in FP8 for a
matmul and read back for a higher-precision accumulation.

This machine has the `ml_dtypes` package installed, which registers
`ml_dtypes.float8_e4m3fn` and `ml_dtypes.float8_e5m2` as real NumPy dtypes —
the same bit layouts used by PyTorch's `torch.float8_e4m3fn` /
`torch.float8_e5m2` and by the hardware itself. You may (and should) use
these dtypes directly via `ndarray.astype(...)`.

## Task

Implement `fp8_round_trip(x, fmt)`.

- `x`: a NumPy array of floats (any float dtype), any shape.
- `fmt`: either the string `"e4m3"` or the string `"e5m2"`, selecting the
  target format.

Quantize every element of `x` into the selected 8-bit floating point format,
then immediately dequantize it back. Return the result as a `float32` NumPy
array of the same shape as `x`. Use `ml_dtypes.float8_e4m3fn` for `"e4m3"`
and `ml_dtypes.float8_e5m2` for `"e5m2"` — these are the exact formats
described above; a differently-biased or infinity-carrying variant will not
match hardware behavior.

## Example

```python
import numpy as np

x = np.array([1.5, 0.001, 100.0, -3.234], dtype=np.float32)
y = fp8_round_trip(x, "e4m3")
# y ~= [1.5, 0.001953125, 96.0, -3.25]   (snapped to the E4M3 grid)
```

## What the gate checks

The gate builds several random `float32` arrays (covering both normal and
near-subnormal magnitudes, well within each format's non-overflowing range)
and, for each, computes the reference round trip directly via
`x.astype(ml_dtypes.float8_e4m3fn).astype(np.float32)` (and the `e5m2`
equivalent) — the real hardware-accurate NumPy dtype cast, not a hand-derived
approximation. It compares your output elementwise against this oracle with

$$
\text{max\_abs\_err} = \max_i \left| y_i - \hat{y}_i \right|
$$

over both formats, and the worst value across all cases must satisfy
$\text{max\_abs\_err} \le 10^{-6}$. Skipping the dequantization step,
quantizing to the wrong format, or picking a differently-biased 8-bit
variant all produce a visibly different grid and fail the gate.
