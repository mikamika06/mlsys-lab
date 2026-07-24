## Context

A `float32` is stored as one sign bit, 8 exponent bits, and 23 mantissa
bits: $[\,s\,|\,e\,|\,m\,]$. This is a **sign-magnitude** encoding, not
two's complement: the low 31 bits, read as a plain unsigned integer, are
already monotonically increasing in $|x|$ as $x$ ranges over the
non-negative floats (including through the subnormals, up through the
largest finite value, into $+\infty$).

That fact makes "the next float toward $+\infty$" a pure integer
operation on the bit pattern — **if** you handle the sign correctly:

* For $x \ge 0$: moving toward $+\infty$ means moving *away* from zero,
  i.e. the 31-bit magnitude increases by 1.
* For $x < 0$: moving toward $+\infty$ means moving *toward* zero, i.e.
  the 31-bit magnitude **decreases** by 1 (a larger magnitude with the
  sign bit set is a *more negative*, i.e. *smaller*, number).
* Special case: $-0.0$ has magnitude $0$; there is nothing smaller to
  decrement to. By IEEE-754, `nextafter(-0, +inf) == nextafter(+0, +inf)`
  — both equal the smallest positive subnormal.

## Task

Implement `next_up`:

```python
def next_up(x: np.ndarray) -> np.ndarray:
    ...
```

* `x` — a `float32` NumPy array. Every element is finite (never `inf` or
  `NaN`), but may include `+0.0`, `-0.0`, subnormals, and the largest
  finite `float32`.
* Returns a `float32` array of the same shape: for each element, the
  next representable `float32` strictly greater than it.

You must compute this via **integer arithmetic on the raw bit pattern**
(e.g. `x.view(np.uint32)`), not by calling `np.nextafter` or doing
floating-point arithmetic on `x` itself.

## Example

```python
import numpy as np
x = np.array([1.0, -1.0, 0.0, -0.0], dtype=np.float32)
print(next_up(x))
# [1.0000001  -0.99999994  1e-45  1e-45]
```

Note how `1.0` and `-1.0` step by *different* absolute amounts (ULP size
depends on magnitude), and both `+0.0` and `-0.0` map to the same value:
the smallest positive subnormal.

## What the gate checks

**exact_match** — the grader builds a fixture of hand-picked corner
cases (±0, ±smallest subnormal, ±largest finite value, ±1, ±0.5) plus
500 random finite `float32` bit patterns, calls `np.nextafter(x, +inf)`
as the oracle, and compares your output to it **bit-for-bit** via the
`uint32` view of both arrays. Fraction of exactly matching bit patterns
must equal `1.0` — no floating-point tolerance is involved, since this
is an exact bit-level identity, not an approximation.
