## Context

An IEEE-754 single-precision float32 occupies 32 bits. The most-significant bit (bit 31) is the **sign bit** $s$. Bits 30–23 form the biased exponent $e$, and bits 22–0 encode the mantissa $m$. For normal numbers the decoded value is

$$
(-1)^{s} \;\times\; 2^{\,e - 127} \;\times\; \left(1 + \frac{m}{2^{23}}\right).
$$

Special encodings (zero, infinity, NaN) also carry a sign bit: positive zero has $s = 0$ while negative zero has $s = 1$, even though both compare equal under `==`.

To inspect the raw bit pattern of a float32 in NumPy, use `.view(np.uint32)` which reinterprets the same 32 bytes as an unsigned integer. The sign bit then lives in the highest-order position and can be isolated with a right-shift:

$$
\text{sign\_bit}(x_i) \;=\; \bigl(\texttt{view\_as\_uint32}(x_i) \gg 31\bigr) \;\in\; \{0, 1\}.
$$

## Task

Implement `extract_sign_bit`:

```python
import numpy as np

def extract_sign_bit(arr: np.ndarray) -> np.ndarray:
    ...
```

The function receives a 1-D NumPy array of `float32` values and must return a **`uint8`** array of the same shape. Element $i$ of the output is `1` if the sign bit of `arr[i]` is set (i.e.\ the value is negative or negative-zero), and `0` otherwise.

Use only vectorized NumPy operations — no Python loops. The result must be `uint8`.

## Example

```python
arr = np.array([1.0, -1.0, 0.0, -0.0,
                np.float32('inf'), np.float32('-inf')],
               dtype=np.float32)
signs = extract_sign_bit(arr)
# array([0, 1, 0, 1, 0, 1], dtype=uint8)
```

Note how $+0.0$ and $-0.0$ both decode to the same numeric value zero, but have **different** sign bits.

## What the gate checks

The gate computes the fraction of identical bytes between your output and a NumPy-oracle reference using `scorers.byte_exact_fraction`. The reference is derived at grading time by the oracle itself — `(arr.view(np.uint32) >> 31).astype(np.uint8)` — so no values are hard-coded. The gate passes when the fraction equals `1.0` (byte-exact match). Any off-by-one bit, wrong dtype, or wrong shape fails the gate.
