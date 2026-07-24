## Context

The IEEE‑754 binary32 format represents a floating‑point number $x$ as

$$x = (-1)^{s}\,2^{e-127}\,\bigl(1.m\bigr),$$

where $s\in\{0,1\}$ is the sign bit, $e\in[0,255]$ is the biased exponent,
and $m\in[0,2^{23}-1]$ are the 23 mantissa bits.  
In a NumPy ``float32`` array each element occupies 4 bytes; viewing it as
``uint32`` exposes the raw bit pattern:

```python
bits = arr.view(np.uint32)
```

The three fields can then be extracted with simple bit‑wise operations:
shift and mask.

## Task

Implement `decompose_floats(arr)` that takes a one‑dimensional NumPy array of
dtype ``float32`` and returns a tuple of three arrays:

```python
signs, exps, mantissas = decompose_floats(arr)
```

* **signs** – unsigned integer array containing the sign bit (0 or 1).  
* **exps** – unsigned integer array containing the biased exponent (0‑255).  
* **mantissas** – unsigned integer array containing the raw mantissa bits
  (0‑$2^{23}-1$).

The implementation must use only NumPy vectorised operations and bitwise
operators; it may not call ``struct.unpack`` or ``np.frexp``. All returned
arrays should have dtype ``uint32`` and match the shape of `arr`.

## Example

```python
import numpy as np
from your_module import decompose_floats

A = np.array([0.0, -1.5, 3.1415927], dtype=np.float32)
signs, exps, mantissas = decompose_floats(A)

print(signs)      # [0 1 0]
print(exps)       # [127 128 129]   (biased exponents)
print(mantissas)  # [0 0x400000 0x921fb6]  (hex for clarity)
```

## What the gate checks

The grader computes a reference decomposition using NumPy’s view and
bit‑masking. It then compares each returned array with the reference via
the scorer `byte_exact_fraction`. All three fractions must be exactly
`1.0`; otherwise the solution fails.

No calls to ``struct.unpack`` or ``np.frexp`` are allowed in a correct
implementation; such calls will not affect the gate but are discouraged
in this task.
