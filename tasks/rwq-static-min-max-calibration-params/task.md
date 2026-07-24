## Context

Asymmetric uint8 quantization maps a float tensor $x$ to an unsigned 8-bit integer using a linear transform with parameters **scale** $s > 0$ and **zero-point** $z \in \mathbb{Z}$:

$$q = \mathrm{clamp}\!\left(\mathrm{round}\!\left(\frac{x}{s}\right) + z,\; 0,\; 255\right)$$

$$\hat{x} = s \cdot (q - z)$$

The quantization range $[0, 255]$ must cover the observed tensor range $[x_{\min}, x_{\max}]$.
Mapping $x_{\min} \to 0$ and $x_{\max} \to 255$ gives the calibration formulas:

$$s = \frac{x_{\max} - x_{\min}}{255}$$

$$z = \mathrm{clamp}\!\left(\mathrm{round}\!\left(\frac{-x_{\min}}{s}\right),\; 0,\; 255\right)$$

When $x_{\min} = x_{\max}$ the tensor is constant, $s = 0$, and the zero-point is undefined;
by convention return $z = 0$.

The `round` here is banker's rounding (round half to even), which both Python's `round()` and
`numpy.round()` implement. The clamp to $[0, 255]$ is necessary because the theoretical
zero-point can fall outside the uint8 range when the tensor's values are all positive
or all negative.

## Task

Implement `calibration_params(tensor)`:

```python
def calibration_params(tensor: np.ndarray) -> tuple[float, int]:
    ...
```

Given a NumPy array `tensor` of arbitrary shape and dtype, return a tuple `(scale, zero_point)`:

- `scale` — a Python `float` equal to $(x_{\max} - x_{\min}) / 255$.
- `zero_point` — a Python `int` equal to $\mathrm{clamp}(\mathrm{round}(-x_{\min} / s),\, 0,\, 255)$.

If the tensor is constant ($x_{\min} = x_{\max}$), return `(0.0, 0)`.

Cast the input to `float64` internally to ensure deterministic results regardless of the
input dtype.

## Example

```python
import numpy as np

t = np.array([-1.0, 0.0, 1.0])
s, z = calibration_params(t)
# s ≈ 0.007843137...  (= 2.0 / 255)
# z = 128              (round(1.0 / s) = round(127.5) = 128 by banker's rounding)

t2 = np.array([1000.0, 2000.0, 3000.0])
s2, z2 = calibration_params(t2)
# s2 ≈ 7.843137...    (= 2000.0 / 255)
# z2 = 0              (clamp(round(-127.5), 0, 255) = clamp(-128, 0, 255) = 0)
```

## What the gate checks

Two gates, both requiring the value $1.0$.

**`scale_ok`**: For each of nine test tensors (constant, single-element, negative-only,
positive-only, symmetric, random normal, arange, etc.) the oracle computes scale from
`float64` min/max. The learner's scale must have relative error $\le 10^{-8}$
on every case. A common mistake is using $256$ instead of $255$ in the denominator.

**`zp_ok`**: The learner's zero-point must match the oracle's zero-point exactly (integer
equality) on every case. Common mistakes: using `int()` truncation instead of
`round()` (banker's rounding), forgetting the clamp, or swapping the sign of $x_{\min}$.
