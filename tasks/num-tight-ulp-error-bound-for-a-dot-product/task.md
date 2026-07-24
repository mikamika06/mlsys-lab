## Context

Floating-point numbers are stored with finite precision. The distance between adjacent representable values depends on the format and is measured in units in the last place (ULP). For float32 arithmetic, the machine epsilon is

$$
\varepsilon = \mathrm{eps}_{32}.
$$

A sequence of rounded operations can accumulate error. A common forward-error model uses

$$
\gamma_n = \frac{n\varepsilon}{1-n\varepsilon},
$$

where $n$ is the number of operations contributing to the accumulated rounding error.

For a naive dot product of vectors $a$ and $b$, the computed result can be compared against a higher precision reference:

$$
\mathrm{rel\_err} =
\frac{|\mathrm{dot}_{32}(a,b)-\mathrm{dot}_{64}(a,b)|}
{|\mathrm{dot}_{64}(a,b)|}.
$$

The value $\gamma_n$ provides a worst-case relative error estimate for a dot product of length $n$.

## Task

Implement `dot_error_bound(n)`:

```python
def dot_error_bound(n: int) -> float:
    ...
```

Return the predicted worst-case relative error bound for a length-$n$ naive float32 dot product.

Use NumPy's float32 machine epsilon and compute

$$
\gamma_n = \frac{n\varepsilon}{1-n\varepsilon}.
$$

Return the result as a Python `float`.

## Example

```python
import numpy as np

bound = dot_error_bound(1000)

expected = (1000 * np.finfo(np.float32).eps) / (
    1 - 1000 * np.finfo(np.float32).eps
)
```

## What the gate checks

The gate computes the reference bound from the same floating-point error model and compares it with the submitted implementation for several vector lengths.

It also measures deterministic float32 dot-product errors against float64 NumPy dot products to verify that the returned bound is a valid scale for observed numerical error. The implementation must return the accumulated $n\varepsilon$ bound rather than a single-operation epsilon.
