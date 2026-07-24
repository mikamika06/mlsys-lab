## Context

Calibrating an activation quantizer from a raw absmax is fragile: one
freak outlier sample can blow up the clip range and waste most of the
quantization grid's resolution on values that almost never occur.
ONNX Runtime's **percentile calibrator** instead calibrates the clip
range from a percentile of the observed `|activation|` distribution —
e.g. clip at the 99.9th percentile instead of the 100th (the true max) —
deliberately letting the rarest, most extreme values saturate in
exchange for finer resolution on everything else. Which percentile is
best is a genuine trade-off (too low clips too much signal, too high
wastes resolution on noise), so it is chosen by sweeping a grid of
candidate percentiles and keeping the one with the lowest reconstruction
MSE against the original, unclipped values.

### The sweep

For a tensor $x$ and a candidate percentile $p$:
$$
\tau(p) = \mathrm{percentile}(|x|,\ p), \qquad
x^{clip}_i = \mathrm{clip}(x_i,\ -\tau(p),\ \tau(p))
$$
$$
s(p) = \tau(p) / q_{max}, \qquad
\widehat{x}_i(p) = \mathrm{clip}\!\big(\mathrm{round}(x^{clip}_i / s(p)),\ -q_{max},\ q_{max}\big)\cdot s(p)
$$
(NumPy percentile convention: linear interpolation between the two
nearest ranks.) The reconstruction error is measured against the
**original, unclipped** $x$:
$$
E(p) = \frac{1}{n}\sum_i \big(x_i - \widehat x_i(p)\big)^2.
$$
Given a finite grid of candidate percentiles $p_1,\dots,p_K$, the
calibrator picks $\arg\min_k E(p_k)$.

## Task

Implement:

```python
def percentile_clip_best(x: np.ndarray, percentile_grid: np.ndarray, qmax: int) -> tuple[int, float]:
    ...
```

* `x` — 1-D array of activation-like values.
* `percentile_grid` — 1-D array of candidate percentiles in `[0, 100]`
  (e.g. `[90, 95, 99, 99.9, 100]`).
* `qmax` — positive int; symmetric quantization uses codes in
  `[-qmax, qmax]`.

Return `(index, mse)`: the index into `percentile_grid` of the percentile
minimizing $E(p)$ above, and that minimum MSE value itself.

## Example

```python
import numpy as np
rng = np.random.default_rng(1)
x = rng.normal(size=1000)
x[[3, 17, 42]] = [5.0, -5.0, 5.0]   # a few rare outliers
idx, mse = percentile_clip_best(x, np.array([90.0, 95.0, 99.0, 99.5, 99.9, 100.0]), qmax=7)
# clipping at 100th percentile (the true max) keeps the outliers exact but
# wastes most of the 15-level grid's resolution on them; an interior
# percentile that lets the outliers saturate gives lower overall MSE.
```

## What the gate checks

* **argmin_index** — your returned index must exactly equal the oracle's
  $\arg\min_k E(p_k)$ (computed with NumPy's percentile + the exact
  clip/quantize/dequantize formulas above) on several random cases, each
  built so the true optimum is an **interior** grid point — neither the
  loosest nor the tightest percentile — so a solution that just returns
  `100` (no clipping) or the tightest percentile will fail.
* **rel_err** — the MSE at *your* chosen index must match the oracle's
  minimum MSE to a relative error $\le 10^{-6}$.
