## Context

`bitsandbytes` uses a non-uniform **dynamic quantization map** for 8-bit
optimizer states instead of a uniform linear grid. The map is built once as a
sorted table of $256$ code values in $[-1, 1]$: a signed exponent-mantissa
layout gives many codes close to zero and few codes near the extremes, unlike
a linear grid which spaces all $256$ codes evenly.

The table is built with `max_exponent_bits = 7` and `total_bits = 8`
(`non_sign_bits = 7`). For $i = 0, \dots, 6$:

$$
\text{fraction\_items}_i = 2^{\,i} + 1, \qquad
\text{boundaries}_i = \operatorname{linspace}(0.1, 1, \text{fraction\_items}_i),
$$

$$
\text{means}_i = \tfrac{1}{2}\left(\text{boundaries}_i[:-1] + \text{boundaries}_i[1:]\right),
\qquad
v_i = 10^{\,i-6} \cdot \text{means}_i .
$$

Both $v_i$ and $-v_i$ are added to the table for every $i$. Finally $0$ and
$1.0$ are appended. Sorting this collection gives exactly $256$ values (the
geometric series $2 + 4 + 8 + \dots + 128 = 254$, plus $\{0, 1.0\}$).

For an input vector $x$, the **dynamic quantizer** scales by the absolute
max, $s = \max_i |x_i|$, snaps $x / s$ to the nearest code in the table, and
dequantizes by multiplying back by $s$. The **linear int8 quantizer** instead
uses a uniform grid of $255$ signed levels:

$$
s = \frac{\max_i |x_i|}{127}, \qquad
q_i = \operatorname{clip}(\operatorname{round}(x_i / s), -127, 127), \qquad
\hat{x}_i = s \, q_i .
$$

On heavy-tailed (log-normal-like) data, most values are small relative to a
few large outliers, so the non-uniform dynamic map — which concentrates
resolution near zero — reconstructs the bulk of the values far more
accurately than the linear grid, at the cost of coarser resolution on the
rare large outliers.

## Task

Implement `dynamic_vs_linear_int8_mse(x)`:

```python
def dynamic_vs_linear_int8_mse(x: np.ndarray) -> tuple[float, float]:
    ...
```

Given a 1-D NumPy array `x` (`float64`, length $\geq 1$, not all zeros):

1. Build the $256$-entry dynamic map table exactly as described above.
2. Quantize/dequantize `x` with the dynamic map (nearest code by value,
   scaled by $\max_i |x_i|$).
3. Quantize/dequantize `x` with the linear int8 scheme above.
4. Return `(mse_dynamic, mse_linear)`, the mean squared reconstruction error
   of each scheme against the original `x`.

Do not change the function name, argument, or return order.

## Example

```python
import numpy as np

rng = np.random.default_rng(0)
mag = rng.lognormal(mean=0.0, sigma=1.2, size=4096)
sign = rng.choice([-1.0, 1.0], size=4096)
x = (mag * sign).astype(np.float64)

mse_dynamic, mse_linear = dynamic_vs_linear_int8_mse(x)
# mse_dynamic is noticeably smaller than mse_linear on this skewed input.
```

## What the gate checks

The gate builds an independent NumPy oracle: it constructs the same
$256$-entry dynamic map, quantizes a fixed log-normal, mixed-sign vector with
both the dynamic map and the linear int8 scheme, and computes both MSEs
directly from NumPy. It compares the submission's `(mse_dynamic, mse_linear)`
to the oracle's via relative error (must be at most $10^{-6}$), and separately
checks that the submission's own `mse_dynamic` is strictly less than its own
`mse_linear` on this skewed input — verifying that the implementation
reproduces the real error advantage of the dynamic map on heavy-tailed data.
