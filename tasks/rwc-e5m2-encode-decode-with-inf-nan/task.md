## Context

The **E5M2** format is one of the two standard 8-bit floating-point (FP8) types introduced in the 2022 FP8 specification. Its bit layout is:

$$\underbrace{s}_{1\text{ bit}} \; \underbrace{e_4 e_3 e_2 e_1 e_0}_{5\text{ bits exponent}} \; \underbrace{m_1 m_0}_{2\text{ bits mantissa}}$$

Unlike E4M3FN (which sacrifices inf/NaN for extra finite range), E5M2 follows IEEE 754 conventions:

- **Exponent bias**: $15$ (for a 5-bit exponent, bias $= 2^{5-1} - 1 = 15$).
- **Normal numbers**: stored exponent $e \in [1, 30]$, value $= (-1)^s \cdot 2^{e-15} \cdot (1 + m/4)$.
- **Subnormal numbers**: stored exponent $e = 0$, value $= (-1)^s \cdot 2^{-14} \cdot (m/4)$.
- **Infinity**: stored exponent $e = 31$, mantissa $= 0$. Represents $\pm\infty$.
- **NaN**: stored exponent $e = 31$, mantissa $\neq 0$. Represents Not-a-Number.
- **Overflow**: values whose magnitude exceeds the maximum finite value ($\approx 57344$) encode as $\pm\infty$.

Rounding uses **round-to-nearest-even** (RNE / banker's rounding).

The maximum finite magnitude is $2^{30-15} \cdot (1 + 3/4) = 2^{15} \cdot 1.75 = 57344$.

## Task

Implement two functions:

```python
def encode_e5m2(values: np.ndarray) -> np.ndarray:
    ...

def decode_e5m2(codes: np.ndarray) -> np.ndarray:
    ...
```

- `encode_e5m2(values)`: Takes a float32 NumPy array and returns a uint8 NumPy array of the same shape, where each element is the E5M2 bit pattern for that value.
  - NaN inputs → NaN code (e.g., `0x7F`, exponent=31, mantissa=3).
  - $\pm\infty$ inputs → $\pm\infty$ codes (`0x7C` / `0xFC`).
  - Overflow (magnitude > 57344) → $\pm\infty$.
  - Round-to-nearest-even for normal/subnormal quantization.
- `decode_e5m2(codes)`: Takes a uint8 NumPy array and returns a float32 array with the decoded values.

## Example

```python
import numpy as np
codes = encode_e5m2(np.array([0.0, 1.0, -2.0, np.inf, np.nan, 1e6], dtype=np.float32))
# codes: [0x00, 0x3C, 0xC0, 0x7C, 0x7F, 0x7C]
vals = decode_e5m2(codes)
# vals: [0., 1., -2., inf, nan, inf]
```

## What the gate checks

The gate encodes an array of special and normal float32 values using the student's `encode_e5m2`, then decodes the result with `decode_e5m2`, and compares both outputs to a reference NumPy-based E5M2 codec. The **exact_match** metric is 1.0 only if every encoded byte and every decoded value (including NaN positions) are identical to the reference.
