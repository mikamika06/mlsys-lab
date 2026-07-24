## Context

FP8 E5M2 is an 8-bit floating-point format used for weights and activations
in low-precision inference/training. Each code is one byte laid out as

$$
\underbrace{s}_{1\text{ bit}}\ \underbrace{e_4 e_3 e_2 e_1 e_0}_{5\text{ bits}}\ \underbrace{m_1 m_0}_{2\text{ bits}}
$$

with exponent bias $15$. A code decodes to a value exactly like IEEE-754,
but with only 5 exponent bits and 2 mantissa bits:

- **Normal** ($1 \le e \le 30$): $v = (-1)^s \cdot 2^{e-15}\left(1 + \frac{m}{4}\right)$
- **Subnormal** ($e = 0$): $v = (-1)^s \cdot 2^{-14}\cdot\frac{m}{4}$
- **Infinity** ($e = 31, m = 0$): $v = (-1)^s \cdot \infty$
- **NaN** ($e = 31, m \ne 0$)

The largest finite magnitude is at $e=30, m=3$: $2^{15}\cdot 1.75 = 57344$.

## Task

Implement `encode_e5m2(values)`:

```python
def encode_e5m2(values: np.ndarray) -> np.ndarray:
    ...
```

`values` is a NumPy array of `float32`. Return a `uint8` array of the same
shape containing the E5M2 code for each element, using **round-to-nearest,
ties-to-even** on the mantissa (the standard IEEE rounding rule — do not
truncate/floor the mantissa).

Overflow must saturate to signed infinity following the *same*
round-to-nearest rule as everywhere else: treat the value that the exponent
field would represent one step past the maximum ($2^{16} = 65536$, i.e. as
if $e=31,m=0$ continued the normal progression) as the neighboring grid
point for rounding purposes, then map that outcome to the infinity code.
Concretely, a magnitude that rounds closer to $65536$ than to the true
maximum finite value $57344$ (or ties, since $65536$'s code parity is even)
becomes $\pm\infty$; otherwise it saturates to $\pm 57344$. The naive rule
"anything above 57344 is immediately $\infty$" is **wrong** — for example
$60000$ is still closer to $57344$ than to the $\infty$ threshold and must
encode to the max-finite code, not the infinity code.

`NaN` inputs must encode to a code with $e=31, m\neq 0$, using the sign bit
of the input. The input array must not be modified.

## Example

```python
import numpy as np

x = np.array([0.0, 1.0, -1.5, 57344.0, 60000.0, 70000.0, np.inf, np.nan], dtype=np.float32)
codes = encode_e5m2(x)
# codes[3] == 0b0_11110_11        (57344, exact)
# codes[4] == 0b0_11110_11        (60000 rounds down to max finite, NOT inf)
# codes[5] == 0b0_11111_00        (70000 overflows to +inf)
```

## What the gate checks

The gate builds the E5M2 decode table directly from the bit-layout formula
above for every one of the 256 possible codes — this *is* the oracle, since
decoding is an unambiguous closed-form mapping. From that table it derives
the correctly-rounded encoder (nearest grid point, ties to even, with the
$65536$-boundary overflow rule described above) and uses it as the reference
for a large batch of test values: every exactly-representable grid value
(so every code must round-trip through `encode_e5m2` to itself), values near
the RNE tie points, values across the subnormal/normal/overflow boundaries,
and explicit `inf`/`nan` inputs.

The returned codes are compared byte-for-byte against the reference codes
using

$$
\text{byte\_exact\_fraction} = \frac{\#\{i : \text{code}_i = \widehat{\text{code}}_i\}}{N}.
$$

The gate requires this fraction to be exactly $1.0$ — every single code
must match. A solution that truncates the mantissa instead of rounding, or
that saturates to infinity too early past $57344$, will mismatch on a
sizeable fraction of the test values and fail the gate.
