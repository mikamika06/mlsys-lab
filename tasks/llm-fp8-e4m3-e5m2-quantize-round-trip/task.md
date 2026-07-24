## Context

FP8 quantization stores floating point values using fewer bits than formats such as
float32. Two common layouts are E4M3 and E5M2, where one bit is the sign bit, the
next $e$ bits store an exponent, and the remaining $m$ bits store a mantissa.

For this task, an FP8 value is defined as a finite-only format with:

$$
x = (-1)^s \times 2^{E-\mathrm{bias}} \times (1 + \frac{M}{2^m})
$$

for normal values, where $E$ is the stored exponent field and $M$ is the mantissa
field. The exponent biases are $7$ for E4M3 and $15$ for E5M2.

Quantization maps a float32 value to the nearest representable FP8 value by
rounding the discarded float32 mantissa bits. Dequantization expands the FP8
bits back into float64 values.

## Task

Implement:

```python
def fp8_roundtrip(x: np.ndarray, fmt: str) -> np.ndarray:
    ...
```

The function receives a float NumPy array and a format name, either `"e4m3"` or
`"e5m2"`. It must emulate FP8 quantization and immediately dequantize the values.
Return a float64 NumPy array with the same shape as `x`.

Use bit-level manipulation of IEEE float32 representations. The implementation
should not call a hardware FP8 kernel or depend on GPU libraries.

## Example

```python
import numpy as np

x = np.array([1.0, 1.5, -2.25], dtype=np.float32)

y = fp8_roundtrip(x, "e4m3")
# y contains the nearest E4M3 representable values as float64
```

## What the gate checks

The gate computes an FP8 round-trip reference using an independent bit
manipulation oracle and compares the submitted function against it.

The metric is

$$
\mathrm{max\_abs\_err} =
\max_i |y_i - y_i^{\mathrm{ref}}|.
$$

The result must satisfy $\mathrm{max\_abs\_err} < 0.2$ for both FP8 formats.
The check uses values that exercise normal numbers, small magnitudes, signs, and
large finite values.
