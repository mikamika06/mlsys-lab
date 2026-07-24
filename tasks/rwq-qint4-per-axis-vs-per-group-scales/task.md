## Context

Symmetric int4 quantization maps a real value $x$ to a signed 4-bit code using a single
scale:

$$
q = \mathrm{clip}\!\left(\mathrm{round}\!\left(\frac{x}{s}\right), -7, 7\right),
\qquad
\hat{x} = q \cdot s, \qquad s = \frac{\max_i |x_i|}{7}.
$$

The scale $s$ is shared by every element it's computed from — the only design choice is
*how many* elements share one scale, i.e. the quantization **granularity**:

- **Per-axis**: one scale per row (per output channel) — $s$ is computed from the
  absolute max of the *entire row*.
- **Per-group**: one scale per contiguous group of `group_size` columns within a row —
  $s$ is computed from the absolute max of *just that group*.

A row whose columns have wildly different magnitude scales (e.g. one segment of small
values, another of large ones) is quantized poorly per-axis: the single row-wide scale
is stretched to cover the largest segment, crushing the smaller ones toward zero. A
per-group scale adapts to each segment separately, so — for a fixed grid size (4 bits)
— finer granularity strictly trades more scale storage for lower reconstruction error.

## Task

Implement `qint4_granularity_mse`:

```python
def qint4_granularity_mse(W: np.ndarray, group_size: int = 32) -> tuple[float, float]:
    ...
```

- `W`: `float64` array of shape `(rows, cols)`, `cols` an exact multiple of
  `group_size`.
- `group_size`: number of columns per per-group scale.

Quantize `W` twice with the symmetric int4 scheme above — once with one scale per row
(per-axis), once with one scale per `group_size`-column group (per-group) — and return
`(mse_per_axis, mse_per_group)`, the mean squared reconstruction error
$\mathrm{mean}((W - \hat{W})^2)$ over the whole matrix for each granularity.

If a row (or group) is all zero, use $s = 1$ instead of $s = 0$ to avoid dividing by
zero (every code in it is then exactly $0$ regardless).

## Example

```python
import numpy as np

# One row, two 4-column groups with very different magnitudes.
W = np.array([[0.01, -0.01, 0.01, -0.01, 10.0, -10.0, 10.0, -10.0]])
mse_axis, mse_group = qint4_granularity_mse(W, group_size=4)
# per-axis: one scale = 10/7 for the whole row -> the 0.01 group is crushed to 0
# per-group: each group gets its own scale -> both groups reconstruct almost exactly
# mse_group << mse_axis
```

## What the gate checks

The gate builds a NumPy oracle that runs the identical per-axis and per-group symmetric
int4 quantization on a fixed test weight matrix whose groups have deliberately different
magnitude scales within each row. It checks:

- `mse_per_axis_err`: absolute error of your `mse_per_axis` vs. the oracle's, at most
  $10^{-9}$.
- `mse_per_group_err`: absolute error of your `mse_per_group` vs. the oracle's, at most
  $10^{-9}$.
- `finer_grain_wins`: your own `mse_per_group` must be strictly less than your own
  `mse_per_axis` (must be `1.0`) — the finer granularity must actually win.
