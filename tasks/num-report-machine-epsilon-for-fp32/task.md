## Context

In IEEE‑754 single precision, a floating‑point number is represented by a sign bit, an 8‑bit exponent field and a 23‑bit fraction (mantissa). The *machine epsilon* $\varepsilon$ for a given format is the smallest positive value that, when added to $1$, yields a result distinguishable from $1$. For single precision this value is

$$\varepsilon = 2^{-23} \approx 1.1920929\times10^{-7}.$$

Its binary representation in 32‑bit IEEE format has the bit pattern

$$
0\,01101000\,00000000000000000000000,
$$

which corresponds to the unsigned integer $0x34000000$.

## Task

Implement `machine_epsilon_fp32()`:

```python
def machine_epsilon_fp32() -> int:
    ...
```

The function should return **the 32‑bit unsigned integer that encodes** the machine epsilon for single precision. Do not return the floating‑point value itself; return its bit pattern as an integer.

## Example

```python
>>> import numpy as np
>>> eps = np.finfo(np.float32).eps
>>> eps.view(np.uint32)
<array(872415232, dtype=uint32)>
>>> machine_epsilon_fp32()
872415232  # same value
```

## What the gate checks

The grader computes the reference bit pattern using NumPy’s `finfo` and compares it with your output. The function must return an integer equal to that reference; any other type or value fails the gate.
