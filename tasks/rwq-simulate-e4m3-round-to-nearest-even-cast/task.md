## Context

FP8 with the **E4M3** format uses 1 sign bit, 4 exponent bits, and 3 mantissa
bits (no inf, but NaN = `S.1111.111`). The finite value set includes:

- **Normal numbers**: exponent $e \in [1, 14]$, value $(-1)^s \cdot 2^{e-7} \cdot (1 + m/8)$
- **Subnormal numbers**: exponent $e = 0$, value $(-1)^s \cdot 2^{-6} \cdot (m/8)$
- Maximum finite magnitude: $448 = 2^{7} \cdot (1 + 7/8)$

When casting a float32 value to E4M3, the standard specifies **round-to-nearest,
ties-to-even**. Values with magnitude exceeding 448 are **clamped** to $\pm 448$
(no infinities in E4M3). NaN inputs map to the E4M3 NaN.

The full representable set has 255 finite values (plus NaN), symmetric around
zero (except subnormals).

## Task

Implement `cast_to_e4m3(x)`:

```python
def cast_to_e4m3(x):
    ...
```

- `x`: float32 NumPy array (any shape).

Return a float32 NumPy array of the same shape containing the nearest E4M3
representable value for each element. Use round-to-nearest-even tiebreaking and
clamp finite inputs to $\pm 448$.

**Algorithm hint**: enumerate all 256 E4M3 bit-patterns, decode each to a float
value, build a value grid, then for each input find the nearest value with
proper tie-breaking.

## Example

```python
import numpy as np
x = np.array([0.0, 1.0, 450.0, -0.002, float('nan')], dtype=np.float32)
y = cast_to_e4m3(x)
# 0.0   -> 0.0
# 1.0   -> 1.0   (exactly representable)
# 450.0 -> 448.0 (clamped)
# -0.002 -> nearest E4M3 subnormal
# nan   -> nan
```

## What the gate checks

The grader enumerates all 256 E4M3 bit-patterns to build the exact value grid,
applies round-to-nearest-even with clamp at $\pm 448$ as the reference oracle,
and compares to your output. The gate passes when every value matches exactly
(`exact_match == 1.0`).
