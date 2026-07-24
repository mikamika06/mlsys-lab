## Context

FP8 E4M3FN is an 8-bit floating point format used in machine learning. It has
one sign bit, four exponent bits, and three fraction bits. The exponent bias is
$7$.

For normal values, the decoded value is

$$
(-1)^s \left(1 + \frac{m}{2^3}\right) 2^{e-7},
$$

where $e$ is the stored exponent and $m$ is the stored fraction. The exponent
field $e=0$ is reserved for subnormal values:

$$
(-1)^s \left(\frac{m}{2^3}\right)2^{-6}.
$$

E4M3FN does not represent infinities. The largest finite value is $448$, and
values outside the range must saturate to $+448$ or $-448$. The bit pattern with
the maximum exponent and maximum fraction is not a finite number and must not be
produced.

Encoding must use round-to-nearest-even (RNE). When two representable values are
equally close, the value whose stored fraction has an even least-significant bit
is selected.

## Task

Implement these functions:

```python
def encode_e4m3fn(x: np.ndarray) -> np.ndarray:
    ...

def decode_e4m3fn(codes: np.ndarray) -> np.ndarray:
    ...
```

`encode_e4m3fn` takes a NumPy floating-point array and returns a `uint8` array
containing E4M3FN bit patterns.

`decode_e4m3fn` takes a `uint8` array of E4M3FN bit patterns and returns a
`float32` NumPy array.

The encoder must perform RNE rounding, handle subnormal values, and saturate
finite overflow to the closest signed value with magnitude $448$.

## Example

```python
import numpy as np

x = np.array([0.0, 1.0, 0.5, 448.0, 1000.0], dtype=np.float32)

codes = encode_e4m3fn(x)
y = decode_e4m3fn(codes)

# y is approximately:
# [0.0, 1.0, 0.5, 448.0, 448.0]
```

## What the gate checks

The gate builds a bit-level E4M3FN oracle and compares the encoded byte output
exactly. The cases include zeros, subnormals, normal values, halfway rounding
cases, values close to $448$, and overflow magnitudes.

The decoded output is compared with the oracle using maximum absolute error:

$$
\max_i |y_i-\hat{y}_i|.
$$

A correct implementation must match the encoded representation and decode the
format accurately.
