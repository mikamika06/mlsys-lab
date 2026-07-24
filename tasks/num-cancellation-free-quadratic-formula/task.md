## Context

The standard quadratic formula solves

$$
ax^2 + bx + c = 0
$$

with

$$
x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}.
$$

When $b$ is large and the discriminant is close to $b^2$, one of the two roots can suffer catastrophic cancellation. For example, subtracting two nearly equal floating-point numbers in

$$
-b + \sqrt{b^2 - 4ac}
$$

can discard many significant digits.

A numerically stable approach computes one root using

$$
q = -\frac{b + \operatorname{sign}(b)\sqrt{b^2 - 4ac}}{2},
$$

then obtains the other root from the product of roots:

$$
x_1 = \frac{q}{a}, \qquad x_2 = \frac{c}{q}.
$$

This avoids subtracting nearly equal quantities and preserves accuracy when the coefficients have very different scales.

## Task

Implement `solve_quadratic(a, b, c)`.

The function receives three real-valued coefficients with $a \ne 0$ and a positive discriminant. Return a tuple `(x1, x2)` containing the two real roots as Python floats.

Use the cancellation-free formulation:

```python
def solve_quadratic(a: float, b: float, c: float) -> tuple[float, float]:
    ...
```

The order of the two roots does not matter.

## Example

```python
roots = solve_quadratic(1.0, 1e8, 1.0)

# The roots are approximately:
# -100000000.0 and -1e-8
```

The implementation should keep both roots accurate even when the naive formula loses precision.

## What the gate checks

The gate compares both returned roots against a high-precision decimal oracle computed from the same quadratic equation. The checker matches roots in either order and reports the maximum relative error.

The required value is

$$
\mathrm{rel\_err} < 10^{-12}.
$$

A direct implementation of the standard formula is expected to fail on cases where $b^2 \gg 4ac$ because one root loses significant digits through cancellation.
