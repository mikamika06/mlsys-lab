## Context

The bfloat16 format keeps the sign bit, the 8 exponent bits, and the top 7 bits
of the FP32 mantissa. An FP32 value has 32 bits:

$$
\mathrm{FP32} = s \; e_7e_6\dots e_0 \; m_{22}m_{21}\dots m_0 .
$$

A bfloat16 value stores the top 16 bits of the FP32 representation. The lower
16 mantissa bits are discarded, but the conversion is not simple truncation.
Round-to-nearest-even (RNE) is applied using the discarded bits.

Let $x$ be the 32-bit integer representation of an FP32 number. The truncated
bfloat16 code is the upper 16 bits:

$$
q = x \gg 16 .
$$

The discarded portion is $r = x \mathbin{\&} 0xffff$. RNE increments $q$ when
the discarded value is greater than half an ulp, or when it is exactly half an
ulp and the retained value has an odd least-significant bit:

$$
q' =
\begin{cases}
q+1 & r > 2^{15} \\
q+1 & r = 2^{15} \land (q \bmod 2)=1 \\
q & \text{otherwise}
\end{cases}
$$

The resulting 16-bit integer is the bfloat16 bit pattern.

## Task

Implement `fp32_to_bf16_codes(values)`:

```python
def fp32_to_bf16_codes(values: np.ndarray) -> np.ndarray:
    ...
```

The input is a NumPy array containing `float32` values. Return a NumPy array
with the same shape and dtype `uint16`, where each element is the bfloat16 bit
pattern produced by round-to-nearest-even conversion.

Do not call external bfloat16 conversion libraries. Implement the bit-level
conversion directly using NumPy operations.

## Example

```python
import numpy as np

x = np.array([1.0, -2.5], dtype=np.float32)
codes = fp32_to_bf16_codes(x)

# codes contains the uint16 representations of the bfloat16 values
# produced by RNE conversion.
```

## What the gate checks

The grader compares the returned bfloat16 codes against the reference conversion
implemented by the `ml_dtypes` bfloat16 dtype. The comparison uses the fraction of
identical bytes:

$$
\mathrm{byte\_exact\_fraction}
=
\frac{\text{number of equal bytes}}{\text{total bytes}} .
$$

The gate requires this value to be exactly $1.0$, so truncation without RNE and
other rounding mistakes do not pass.
