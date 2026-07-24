## Context

`x.astype(np.float32)` on a `float64` array performs a hardware-correct
round-to-nearest-even conversion. This task asks you to reproduce that cast
by hand, at the bit level, using the classic **guard/round/sticky (GRS)**
technique used by floating-point hardware.

A `float64` has 1 sign bit, 11 exponent bits (bias 1023), and 52 mantissa
bits. A `float32` has 1 sign bit, 8 exponent bits (bias 127), and 23 mantissa
bits. Converting means keeping the top 23 of the 52 mantissa bits and
correctly rounding away the bottom 29:

$$
\text{mantissa}_{64} = \underbrace{m_{51}\dots m_{29}}_{\text{kept, 23 bits}}\;
\underbrace{m_{28}\; m_{27}\dots m_{0}}_{\text{dropped, 29 bits}}
$$

Of the 29 dropped bits, call $g = m_{28}$ the **guard** bit, and let $r$ be
the logical OR of the remaining 28 bits (the traditional round+sticky
combined). Treat the dropped 29-bit field as a single unsigned integer $d$
with half-a-ulp $h = 2^{28}$. Round-to-nearest-even then says:

$$
\text{round\_up} =
\begin{cases}
\text{true} & d > h \\
\text{true} & d = h \ \text{and kept mantissa's LSB} = 1 \\
\text{false} & \text{otherwise}
\end{cases}
$$

If rounding the 23-bit kept mantissa up overflows (`0x7FFFFF + 1 =
0x800000`), the mantissa wraps to 0 **and the exponent must be incremented
by one** — exactly as a carry propagates in ordinary addition.

## Task

Implement `fp64_to_fp32_bits(x)`:

```python
def fp64_to_fp32_bits(x: np.ndarray) -> np.ndarray:
    ...
```

- `x` is a NumPy array of `float64` values (any shape), finite, nonzero, and
  chosen so that both the input and the rounded output stay comfortably
  within the normal (non-subnormal, non-overflowing) exponent range of
  `float32` — you do not need to handle subnormals, zero, infinities, or
  NaN.
- Return a NumPy array of dtype `uint32`, same shape as `x`, holding the
  bit pattern of the `float32` result of casting `x`, computed by hand from
  the `float64` bit fields (sign, exponent, mantissa) using GRS
  round-to-nearest-even as described above — not by calling
  `x.astype(np.float32)` or any other library conversion.

## Example

```python
import numpy as np

x = np.array([1.0, 3.14159265358979, -2.5], dtype=np.float64)
bits = fp64_to_fp32_bits(x)

# bits[i] equals x[i].astype(np.float32).view(np.uint32), computed here
# by hand via guard/round/sticky rounding instead of a library cast.
```

## What the gate checks

The grader builds a large batch of random `float64` values spanning many
exponents and mantissa patterns, plus three hand-constructed edge cases that
isolate the tricky rounding paths: an exact tie that must round **down**
(kept mantissa already even), an exact tie that must round **up** to reach
an even mantissa, and a rounding step that **overflows the mantissa and
carries into the exponent**.

For every input, the grader compares your `uint32` output against the real
oracle `x.astype(np.float32).view(np.uint32)`, byte for byte, via
`byte_exact_fraction`. The gate requires this fraction to be exactly `1.0` —
any missed tie-to-even case or missing carry-into-exponent handling produces
a mismatch and fails the gate.
