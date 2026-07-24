## Context

HQQ (Half-Quadratic Quantization) quantizes a weight tensor without any
calibration data by directly optimizing scale and zero-point to minimize a
robust loss. Before that optimization loop runs, HQQ needs a starting point:
a plain closed-form **asymmetric min/max group quantizer**.

For a group $g$ of `group_size` weight values and a bit width $n$:

$$
s = \frac{\max(g) - \min(g)}{2^n - 1}, \qquad
z = \operatorname{round}\!\left(\frac{-\min(g)}{s}\right)
$$

Each value $x \in g$ is then mapped to an integer code

$$
q = \operatorname{clip}\!\left(\operatorname{round}\!\left(\frac{x}{s}\right) + z,\ 0,\ 2^n - 1\right)
$$

and dequantized as

$$
\hat{x} = (q - z) \cdot s.
$$

This affine (scale + zero-point) scheme is exact at the group's min and max:
the smallest value in the group rounds to code $0$ and the largest rounds to
code $2^n - 1$, which is exactly why it's a good initializer for the
zero-point refinement that HQQ performs afterward.

## Task

Implement `hqq_init(W, group_size=64, nbits=4)`.

`W` is a NumPy array of any shape. Ravel it in row-major (`C`) order and
split it into consecutive groups of `group_size` elements (the last group
may be shorter if the total size doesn't divide evenly). For every group:

1. Compute `scale = (max(g) - min(g)) / (2**nbits - 1)`. If the group is
   constant (`max(g) == min(g)`), use `scale = 1.0` instead (to avoid
   division by zero).
2. Compute `zero = round(-min(g) / scale)`.
3. Quantize: `code = clip(round(g / scale) + zero, 0, 2**nbits - 1)`, stored
   as `uint8`.
4. Dequantize: `dequant = (code - zero) * scale`.

Return a 4-tuple:

```python
(W_q, scale, zero, dequant)
```

- `W_q`: `uint8` array, same shape as `W`, codes in `[0, 2**nbits - 1]`.
- `scale`: `float64` array of shape `(n_groups,)`, one entry per group.
- `zero`: `float64` array of shape `(n_groups,)`, one entry per group.
- `dequant`: `float64` array, same shape as `W`.

## Example

```python
import numpy as np

W = np.array([0.0, 1.5, -2.5, 7.0, 3.0, -1.0])
W_q, scale, zero, dequant = hqq_init(W, group_size=2, nbits=4)
# group 0 = [0.0, 1.5] -> scale=0.1, zero=0, codes=[0, 15]
# group 1 = [-2.5, 7.0] -> spans that group's own min/max
# group 2 = [3.0, -1.0] -> spans that group's own min/max
```

## What the gate checks

The gate rebuilds the same per-group asymmetric min/max quantizer with an
independent NumPy oracle across several shapes, group sizes, and bit widths
(including an all-constant group and a group size that doesn't evenly divide
the tensor).

- `codes_exact`: every returned code must exactly match the oracle's code
  (integer equality, and every code must lie in `[0, 2**nbits - 1]`).
- `max_abs_err`: the maximum absolute error between your `dequant` output
  and the oracle's dequantized reconstruction must be at most `1e-6`.

A solution that computes `zero` from a different formula (e.g. omitting the
`round`, or using `min` instead of `-min`), or that groups along the wrong
axis, produces different codes and fails `codes_exact`.
