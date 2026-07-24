## Context

IEEE-754 single precision floating point numbers are stored as a sign bit, an exponent field, and a fraction field. A normal fp32 value can be reconstructed as

$$
(-1)^s \left(1 + \frac{f}{2^{23}}\right) 2^{e-127},
$$

where $s$ is the sign bit, $e$ is the stored exponent, and $f$ is the 23-bit fraction.

Floating point addition is not exact integer addition. The significands must first be aligned by exponent. Bits shifted out during alignment are summarized by guard, round, and sticky bits:

$$
\mathrm{GRS} = (g, r, s).
$$

For round-to-nearest-even, the retained significand is incremented when the discarded part is more than half an ulp, or exactly half an ulp and the retained least-significant bit is odd.

The implementation exercise is to reproduce fp32 addition using integer fields and explicit reconstruction of the final bit pattern.

## Task

Implement `fp32_add_bits(a_bits, b_bits)`:

```python
def fp32_add_bits(a_bits: int, b_bits: int) -> int:
    ...
```

The inputs are 32-bit integers containing the raw IEEE-754 fp32 bit patterns. The function must return the raw 32-bit integer bit pattern of the correctly rounded fp32 sum.

Implement the addition pipeline with integer operations:

1. Decode sign, exponent, and fraction fields.
2. Align significands using exponent differences.
3. Add or subtract according to the signs.
4. Normalize the result.
5. Round using guard, round, and sticky bits with round-to-nearest-even.
6. Reconstruct the final fp32 representation.

The function must not convert the inputs to Python `float` or call NumPy floating point addition.

## Example

```python
def bits(x):
    import struct
    return struct.unpack("<I", struct.pack("<f", x))[0]

a = bits(1.5)
b = bits(2.25)

out = fp32_add_bits(a, b)
# out is the bit pattern for 3.75
```

## What the gate checks

The gate computes the reference answer using NumPy fp32 addition and compares the returned bit patterns byte-for-byte.

The `byte_exact_fraction` score is

$$
\frac{\text{number of equal output bytes}}{\text{total output bytes}}.
$$

The required score is $1.0$, so every tested addition must match the real fp32 rounding result exactly. The cases include random finite operands and adversarial exponent-alignment cases that exercise guard, round, and sticky bit handling.
