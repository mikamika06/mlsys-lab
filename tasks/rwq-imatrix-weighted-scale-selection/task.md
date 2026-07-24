## Context

llama.cpp's `imatrix` quantization does not pick a block's quantization
scale by minimizing plain reconstruction MSE. Instead it uses an
**importance matrix**: per-weight sensitivity scores collected by running
calibration text through the model and recording each weight's squared
activation contribution. A weight that rarely fires (low importance)
can be reconstructed sloppily for free; a weight that fires constantly
(high importance) must be reconstructed precisely even if that means a
huge, rarely-hit outlier gets clipped harder. Concretely, instead of
choosing the scale $s$ that minimizes
$\sum_i (x_i - \widehat{x}_i(s))^2$, imatrix quantization minimizes the
**importance-weighted** objective
$\sum_i w_i\,(x_i - \widehat{x}_i(s))^2$
for per-weight importance weights $w_i \ge 0$. Because the two
objectives weight the same errors differently, they can (and often do)
pick a *different* optimal scale for the same block.

### Setup

For a block $x \in \mathbb{R}^n$, symmetric quantization at scale $s>0$
with integer bounds $[q_{min}, q_{max}]$ is
$$
q_i(s) = \mathrm{clip}\big(\mathrm{round}(x_i / s),\ q_{min},\ q_{max}\big),
\qquad
\widehat{x}_i(s) = q_i(s)\cdot s.
$$

Given a finite candidate grid of scales $s_1, \dots, s_K$ and importance
weights $w_1,\dots,w_n \ge 0$, the imatrix-weighted objective for
candidate $s_k$ is
$$
E(s_k) = \sum_{i=1}^n w_i\,\big(x_i - \widehat x_i(s_k)\big)^2,
$$
and the selected scale is $\arg\min_k E(s_k)$ (smallest index on a tie).

## Task

Implement:

```python
def imatrix_best_scale(x: np.ndarray, w: np.ndarray, scale_grid: np.ndarray, qmin: int, qmax: int) -> int:
    ...
```

* `x` — 1-D block of weights.
* `w` — 1-D array of the same length, nonnegative importance weights.
* `scale_grid` — 1-D array of candidate positive scales $s_1,\dots,s_K$.
* `qmin`, `qmax` — integer quantization code bounds (symmetric quant, no
  zero-point).

Return the integer index $k$ into `scale_grid` minimizing $E(s_k)$ above
(the first such index if there is a tie).

## Example

```python
import numpy as np
x = np.array([0.1, 0.11, -0.09, 3.0])       # index 3 is a rare outlier
w = np.array([10.0, 10.0, 10.0, 0.02])      # imatrix says index 3 barely matters
grid = np.array([0.02, 0.05, 0.4])
k = imatrix_best_scale(x, w, grid, qmin=-8, qmax=7)
# the fine scale (0.02) wins: it reproduces the three important
# small values almost exactly and lets the down-weighted outlier clip hard.
# The *unweighted* MSE-optimal scale would instead be the coarse one (0.4),
# which better fits the large outlier at the expense of the small values.
```

## What the gate checks

* **argmin_index** — your returned index must exactly equal
  `argmin_k E(s_k)` computed by a NumPy oracle that quantizes/dequantizes
  every grid candidate and sums the importance-weighted squared error, on
  several random blocks (each with one heavily down-weighted outlier, so
  the weighted argmin provably differs from the plain-MSE argmin — a
  solution that ignores `w` and minimizes plain MSE will fail this gate).
* **rel_err** — the weighted error $E$ at *your* chosen index must match
  the oracle's minimum $E$ to a relative error $\le 10^{-6}$ (catches an
  off-by-one or a formula bug that happens to still return a valid index).
