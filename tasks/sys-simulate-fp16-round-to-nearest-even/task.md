## Context

Low precision machine learning formats often reduce storage and computation cost by
representing numbers with fewer bits. IEEE 754 binary16, commonly called fp16, uses
$1$ sign bit, $5$ exponent bits, and $10$ fraction bits.

A normalized binary floating point value has the form

$$
(-1)^s \times (1.f)_2 \times 2^{e-b},
$$

where $s$ is the sign bit, $f$ is the fraction field, $e$ is the stored exponent,
and $b$ is the exponent bias.

Converting from fp32 to fp16 requires discarding lower precision bits. The required
rounding mode is round-to-nearest-even (RNE). If the discarded part is exactly halfway
between two representable fp16 values, the result is rounded so that the remaining
least significant bit is even. This avoids systematic upward or downward bias.

For a positive finite value, if the discarded bits represent a value greater than
half an fp16 unit, the kept significand is incremented. If they represent exactly half,
the increment happens only when the kept significand is odd.

## Task

Implement `fp32_to_fp16_rne(x)`:

```python
def fp32_to_fp16_rne(x: np.ndarray) -> np.ndarray:
    ...
```

The input is a NumPy array of dtype `float32`. Return a NumPy array of dtype
`float16` containing the same values as an IEEE fp16 conversion using
round-to-nearest-even.

Implement the conversion logic manually by inspecting fp32 bit patterns. Do not use
NumPy's direct dtype conversion helpers such as `astype(np.float16)`. The returned
array must have the same shape as the input.

The implementation should correctly handle normal values, subnormal values, zeros,
infinities, and NaNs.

## Example

```python
import numpy as np

x = np.array([1.5, 1.0009765625, 0.0], dtype=np.float32)
y = fp32_to_fp16_rne(x)

# y has dtype float16 and represents:
# [1.5, 1.001, 0.0]
```

## What the gate checks

The gate computes the reference conversion by using NumPy's IEEE fp16 cast as the
numeric oracle. The returned values are compared against this oracle with
$\mathrm{max\_abs\_err}$.

The gate also rejects solutions that call NumPy's direct `astype` conversion path.
A correct implementation must reproduce the fp16 round-to-nearest-even behavior
through explicit bit manipulation.
