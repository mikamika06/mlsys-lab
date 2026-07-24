## Context

A floating-point cast changes the representation of a real value by reducing the
available precision. IEEE 754 binary32 has fewer mantissa bits than binary64, so
conversion requires rounding.

The preferred rounding mode for most numerical workloads is round-to-nearest,
ties-to-even (RNE). If the discarded part is exactly halfway between two
representable values, the least significant retained bit is chosen to be even.
For a value $x$ with two neighboring representable results $a$ and $b$,

$$
\mathrm{round}_{\mathrm{RNE}}(x)=
\begin{cases}
a, & |x-a| < |x-b|,\\
b, & |x-b| < |x-a|,\\
\text{the value with an even least significant bit}, & |x-a|=|x-b|.
\end{cases}
$$

A broken low-level cast implementation can simply truncate discarded mantissa
bits. Truncation always moves toward the lower representable bit pattern for
positive values and introduces a systematic bias. The fix is to implement a
cast that matches the IEEE 754 round-to-nearest-even result.

## Task

Debug `cast_f32_rne(values)` so that it returns the same binary32 values that
NumPy produces for a float64-to-float32 conversion.

The function signature is:

```python
def cast_f32_rne(values: np.ndarray) -> np.ndarray:
    ...
```

The input is a NumPy array with dtype `float64`. Return a NumPy array with dtype
`float32`. The returned bytes must exactly match the IEEE 754 binary32 encoding
from NumPy's conversion.

The implementation must handle normal values, subnormal values, negative values,
and values exactly on rounding boundaries.

## Example

```python
import numpy as np

x = np.array([1.0, 1.0000000596046448], dtype=np.float64)
y = cast_f32_rne(x)

# y is identical to:
# x.astype(np.float32)
```

## What the gate checks

The gate computes the oracle result with NumPy's real float64-to-float32 cast
and compares the returned byte representation using
$\mathrm{byte\_exact\_fraction}$.

The score must be exactly $1.0$, meaning every output byte matches the NumPy
reference. Boundary values are included because truncation and ties-to-even
produce different binary32 encodings there.
