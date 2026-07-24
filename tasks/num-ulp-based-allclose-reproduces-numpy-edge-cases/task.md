## Context

Floating-point numbers are not evenly distributed. Around a value $x$, the distance
to the next representable floating-point number is given by the spacing function
$\mathrm{spacing}(x)$. Machine precision therefore depends on magnitude: a fixed
absolute tolerance can be too strict for large values and too permissive for values
near zero.

A ULP (unit in the last place) measures this spacing. Two values can be considered
close when their difference is no larger than a number of representable steps:

$$
|a-b| \leq k \cdot \max(\mathrm{spacing}(|a|), \mathrm{spacing}(|b|)),
$$

where $k$ is the allowed number of ULPs.

NumPy's `allclose` uses an absolute and relative tolerance rule:

$$
|a-b| \leq \mathrm{atol} + \mathrm{rtol}|b|.
$$

This rule is useful, but a single `atol` value behaves differently for tiny and
huge magnitudes. A ULP-based comparison can expose cases where an absolute
tolerance accepts values that are many representable steps apart, or rejects
values that are only a few steps apart.

## Task

Implement `ulp_allclose_report(a, b, max_ulps, atol)`:

```python
def ulp_allclose_report(
    a: np.ndarray,
    b: np.ndarray,
    max_ulps: int,
    atol: float
) -> tuple[np.ndarray, np.ndarray]:
    ...
```

The inputs are one-dimensional NumPy arrays of `float64` values with equal length.

Return two boolean arrays:

1. The first array contains the ULP-based verdict for every pair. A pair passes
   when the absolute difference is within `max_ulps` spacings of the values.
2. The second array contains the absolute-tolerance verdict for every pair:
   `abs(a - b) <= atol`.

Use NumPy operations rather than Python loops. The returned arrays must have
boolean dtype and preserve input order.

## Example

```python
import numpy as np

a = np.array([1.0, 1e20, 0.0])
b = np.array([
    np.nextafter(1.0, np.inf),
    1e20 + 1e6,
    1e-20,
])

ulp_ok, atol_ok = ulp_allclose_report(a, b, 2, 1e-8)

# ulp_ok reports closeness in representable steps.
# atol_ok reports closeness using the fixed absolute tolerance.
```

## What the gate checks

The gate generates edge cases containing tiny values, values near zero, and very
large magnitudes. It computes the expected result using NumPy floating-point
spacing operations and compares both returned boolean arrays exactly.

The `exact_match` metric must be $1.0$. A solution that only applies an absolute
tolerance will fail because the two verdict arrays differ on floating-point
spacing edge cases.
