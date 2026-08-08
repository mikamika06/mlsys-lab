# FP4 Microscaling: E2M1 Value Set Enumeration

In this exercise, you will implement the standard `E2M1` (2 exponent bits, 1 mantissa bit) 4-bit floating point format. This fundamental layout is the basis for both OCP Microscaling Formats (MXFP4) and NVIDIA's NVFP4 architectures for ultra-low-precision AI inference and training. Because hardware vendors handle special edge cases differently, your implementation must be parameterized.

A 4-bit E2M1 value has 4 bits laid out as `S E E M`:
- Bit 3 (MSB): Sign (`S`). 0 is positive, 1 is negative.
- Bits 1 and 2: Exponent (`E`). Ranges from 0 to 3.
- Bit 0 (LSB): Mantissa (`M`). 0 or 1.

## Decoding Rules
Given a configuration `bias` (e.g., 1), `has_nan` (bool), and `has_inf` (bool), decode the integer (0 to 15) to a float as follows:
1. If `E == 3`:
   - If `has_nan` is True and `M == 1`, return `NaN` (use `float('nan')`).
   - If `has_inf` is True and `M == 0`, return `Infinity` (signed by `S`).
2. If `E == 0`:
   - If `M == 0`, the value is exactly `0.0` (signed by `S`).
   - If `M == 1`, the value is a **subnormal**: `(-1)^S * 2^(1 - bias) * 0.5`.
3. For all other cases (normals):
   - The value is `(-1)^S * 2^(E - bias) * (1.0 + M * 0.5)`.

## Quantization
To cast a higher-precision tensor down to FP4, you must find the closest *finite* representable value in the format's set.
- Ignore `NaN` and `Inf` when finding the closest quantization bin.
- If an input is exactly halfway between two finite representable values, round to the one with the larger absolute magnitude (round away from zero).

In Milestone 3, write regression tests ensuring strict invariants hold, such as ensuring that `0.0` is always exactly representable.
