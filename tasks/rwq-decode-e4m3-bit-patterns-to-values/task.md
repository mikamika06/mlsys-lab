## Context

The E4M3 format is an 8-bit floating-point type widely adopted in deep-learning
hardware and libraries (e.g. NVIDIA H100 FP8, PyTorch `float8_e4m3fn`). It packs
a sign bit, four biased exponent bits, and three mantissa bits into a single byte:

| Bit 7 | Bits 6–3 | Bits 2–0 |
|-------|----------|----------|
| Sign  | Exponent | Mantissa |

The exponent bias is $7$. There are **no** special infinity or NaN encodings in
E4M3 — every bit pattern represents a finite real number.

**Normalized** ($E \neq 0$):

$$v = (-1)^{S} \;\times\; 2^{\,E-7} \;\times\; \bigl(1 + \tfrac{M}{8}\bigr)$$

**Subnormal** ($E = 0$, $M \neq 0$):

$$v = (-1)^{S} \;\times\; 2^{\,1-7} \;\times\; \frac{M}{8}$$

**Zero** ($E = 0$, $M = 0$):

$$v = (-1)^{S} \times 0$$

The 256-entry lookup table that these formulas produce is deterministic and can
serve as a production-quality oracle for decoding arbitrary bit patterns.

## Task

Implement `decode_e4m3(codes)`:

```python
def decode_e4m3(codes) -> np.ndarray:
    ...
```

`codes` is a 1-D NumPy array of `uint8` values (length $n$, any $n \geq 1$).
Return a 1-D `float64` array of length $n$ where each element is the E4M3
decoded value of the corresponding code.

No Python loops are required — build the 256-entry lookup table once with
vectorized NumPy, then use **fancy indexing** to map every code at once.

## Example

```python
import numpy as np
codes = np.array([0, 1, 127, 128, 255], dtype=np.uint8)
vals = decode_e4m3(codes)
# 0   ->  0.0
# 1   ->  0.001953125   (subnormal, M=1)
# 127 ->  448.0          (E=15, M=7: 2^8 * 1.875)
# 128 -> -0.0           (sign bit set, M=0, E=0)
# 255 -> -448.0
```

## What the gate checks

The gate builds its own reference by applying the E4M3 field formulas to all 256
possible `uint8` codes (0–255) using a separate vectorised NumPy oracle, then
compares every decoded value from the learner's function against that oracle.

The check passes (`exact_match == 1.0`) only when **all** 256 entries match
exactly (NaN patterns, signed zeros, and subnormals included). Any
implementation error in the field decoding — wrong bias, missing subnormal
handling, swapped sign — will cause at least one mismatch.
