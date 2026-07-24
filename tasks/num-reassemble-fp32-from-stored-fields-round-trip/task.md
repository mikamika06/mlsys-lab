## Context

The IEEE‑754 binary32 format represents a floating‑point number $x$ as

$$x = (-1)^{s}\,2^{e-127}\,\bigl(1.f\bigr),$$

where $s\in\{0,1\}$ is the sign bit, $e\in[0,255]$ is the biased exponent,
and $f\in[0,\,1)$ is encoded by a 23‑bit fraction (mantissa).  
The underlying 32 bits are laid out as

$$
\text{bits} = s\,\underbrace{11111111}_{8}\,\underbrace{00000000000000000000000}_{23}.
$$

Reassembling a `float32` from its three fields therefore amounts to packing the
three integers into a 32‑bit unsigned integer and interpreting that as a
floating‑point number.

## Task

Implement `reassemble_fp32(signs, exps, mantissas)`:

```python
def reassemble_fp32(signs: np.ndarray,
                    exps: np.ndarray,
                    mantissas: np.ndarray) -> np.ndarray:
    ...
```

* `signs`   – array of sign bits (0 or 1), any integer dtype.
* `exps`    – array of biased exponents, any integer dtype.
* `mantissas` – array of 23‑bit fraction values, any integer dtype.

All three arrays have the same shape.  
The function must return a NumPy array of type `float32` with that shape,
containing the numbers reconstructed from the supplied fields.

## Example

```python
import numpy as np

# original data
x = np.array([1.5, -2.0, 0.0], dtype=np.float32)

# decompose into fields
bits = x.view(np.uint32)
signs      = (bits >> 31) & 1
exps       = (bits >> 23) & 0xFF
mantissas  = bits & 0x7FFFFF

# reassemble
y = reassemble_fp32(signs, exps, mantissas)

print(y)
# [ 1.5 -2.   0.]
```

## What the gate checks

The grader computes the byte‑wise exact fraction between the output of your
function and a reference reconstruction performed with NumPy’s bit‑level
operations. The metric `byte_exact_fraction` must equal **1.0** for the task to pass.

Additionally, the solution should correctly handle all valid binary32 values,
including subnormals, zeros, infinities, and NaNs.
