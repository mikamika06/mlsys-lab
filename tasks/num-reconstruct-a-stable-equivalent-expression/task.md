## Context

Floating-point arithmetic can lose significant digits when subtracting two nearly equal numbers. This is called catastrophic cancellation.

Consider the expression

$$
f(x) = \frac{1-\cos(x)}{x^2}.
$$

For small $x$, the values of $1$ and $\cos(x)$ are extremely close. Computing the numerator directly can round both values to the same floating-point number, producing an inaccurate result.

Using trigonometric identities, the same mathematical quantity can be written without the problematic subtraction:

$$
1-\cos(x)=2\sin^2\left(\frac{x}{2}\right),
$$

which gives the stable equivalent expression

$$
f(x)=\frac{1}{2}\left(\frac{\sin(x/2)}{x/2}\right)^2 .
$$

This form preserves useful precision for very small inputs.

## Task

Implement `stable_one_minus_cos_over_x2(x)`:

```python
def stable_one_minus_cos_over_x2(x: np.ndarray) -> np.ndarray:
    ...
```

The function receives a NumPy array of floating-point values and returns the same-shaped array containing

$$
\frac{1-\cos(x)}{x^2}
$$

evaluated using a numerically stable equivalent expression. The output must be `float64`.

Use NumPy operations only. Do not compute the numerator as `1 - np.cos(x)` because the inputs are chosen to expose cancellation.

## Example

```python
import numpy as np

x = np.array([1e-2, 1e-8])
y = stable_one_minus_cos_over_x2(x)

# values are close to 0.5
# [0.49999792 0.5]
```

## What the gate checks

The gate compares the returned values against a high-precision-style oracle computed from the stable mathematical form using NumPy `longdouble` arithmetic.

The relative error

$$
\mathrm{rel\_err} =
\frac{\lVert y_{\mathrm{candidate}}-y_{\mathrm{oracle}}\rVert_2}
{\lVert y_{\mathrm{oracle}}\rVert_2+10^{-12}}
$$

must satisfy $\mathrm{rel\_err} \le 10^{-13}$.

Inputs include tiny values where the direct expression loses precision.
