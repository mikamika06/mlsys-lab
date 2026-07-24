## Context

`E2M1` is the 4-bit floating-point format used by OCP Microscaling (MX) FP4:
1 sign bit, 2 exponent bits, 1 mantissa bit. Its 8 representable magnitudes
(the format has no infinities or NaNs) are the fixed grid

$$
G = \{0,\ 0.5,\ 1,\ 1.5,\ 2,\ 3,\ 4,\ 6\} .
$$

Quantizing a value $x$ to E2M1 means finding the nearest magnitude in $G$ to
$|x|$ and pairing it with the sign of $x$. The 4-bit code that a real
quantizer would pack into memory is

$$
\mathrm{code}(x) = 8 \cdot [x < 0] + \operatorname*{arg\,min}_{i \in \{0,\dots,7\}} \bigl| \,|x| - G_i \,\bigr| ,
$$

with ties (an $x$ exactly halfway between two grid points) broken toward the
smaller-magnitude code — the index numpy's `argmin` returns naturally, since
it keeps the first minimum it encounters and $G$ is sorted ascending.

## Task

Implement `e2m1_classify` in `solve.py`:

```python
def e2m1_classify(x: np.ndarray) -> np.ndarray:
    ...
```

* `x` — real-valued array, any shape.

Return an integer array of the same shape as `x`, each entry the signed
4-bit E2M1 code $\mathrm{code}(x) \in \{0, \dots, 15\}$ for the
corresponding element: `sign_bit * 8 + magnitude_index`, where
`magnitude_index` is the index into $G$ (`[0, 0.5, 1, 1.5, 2, 3, 4, 6]`) of
the nearest grid value to `abs(x)`, and `sign_bit` is `1` if `x < 0` else
`0`. Values larger than the largest grid point (`6`) simply classify to the
nearest grid point, which is `6` — there is no separate clipping step.

## Example

```python
import numpy as np

x = np.array([0.0, 0.26, -1.2, 5.9, -100.0])
codes = e2m1_classify(x)
# 0.00 -> nearest grid value 0.0   -> magnitude index 0, sign 0 -> code 0
# 0.26 -> nearest grid value 0.5   -> magnitude index 1, sign 0 -> code 1
# -1.2 -> nearest grid value 1.5   -> magnitude index 3, sign 1 -> code 11
# 5.9  -> nearest grid value 6.0   -> magnitude index 7, sign 0 -> code 7
# -100 -> nearest grid value 6.0 (saturates) -> magnitude index 7, sign 1 -> code 15
```

## What the gate checks

The grader loads the fixture `fp4_x.npy` -- a mix of exact grid points,
exact midpoints between adjacent grid points (the tie-break cases), values
that saturate past the largest magnitude, near-zero values, and random
continuous values, all with both signs -- and builds its own NumPy oracle
using the same nearest-grid-point rule. It also runs two additional
self-generated cases (a random 2D array and a handful of exact grid values
including `+0.0`/`-0.0`). `exact_match` requires your codes to equal the
oracle's on every element of every case.
