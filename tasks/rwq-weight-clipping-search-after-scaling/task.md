## Context

AWQ-style weight quantization first computes a per-channel activation-aware scale, then
runs an additional **clip search**: for every quantization group it tries a small grid
of clipping ratios and keeps whichever ratio produces the lowest reconstruction error.
Clipping shrinks the range used to set the quantization scale, sacrificing accuracy on a
group's rare outlier elements in exchange for a finer step size on everything else in
the group — a trade that often lowers overall MSE.

For a group of weights $w_1, \dots, w_g$ with $a = \max_i |w_i|$, and a candidate ratio
$r \in (0, 1]$:

$$
a_r = a \cdot r, \qquad
s_r = \frac{a_r}{q_{\max}}, \qquad
q_{\max} = 2^{\text{bits}-1} - 1 .
$$

Each element is first hard-clipped to $[-a_r, a_r]$, then quantized/dequantized with
scale $s_r$:

$$
\tilde{w}_i = \mathrm{clip}(w_i, -a_r, a_r), \qquad
q_i = \mathrm{clip}\!\left(\mathrm{round}\!\left(\frac{\tilde{w}_i}{s_r}\right), -q_{\max}, q_{\max}\right),
\qquad
\hat{w}_i(r) = q_i \cdot s_r .
$$

The error at ratio $r$ is measured against the **original, unclipped** group:

$$
\mathrm{MSE}(r) = \frac{1}{g}\sum_i \left(w_i - \hat{w}_i(r)\right)^2 .
$$

The search picks, per group, the ratio (equivalently, its index in the shared candidate
grid) that minimizes $\mathrm{MSE}(r)$.

## Task

Implement `awq_clip_search`:

```python
def awq_clip_search(
    W: np.ndarray, group_size: int, clip_ratios: np.ndarray, bits: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    ...
```

- `W`: `float64` array of shape `(rows, cols)`, `cols` an exact multiple of
  `group_size`.
- `group_size`: number of columns per quantization group.
- `clip_ratios`: 1-D array of candidate ratios, e.g. `np.linspace(1.0, 0.5, 11)`.
- `bits`: bit width for the symmetric quantization grid (default 4,
  $q_{\max} = 2^{\text{bits}-1}-1 = 7$).

For every group, sweep every ratio in `clip_ratios` (in the given order — ties broken by
the earliest/smallest index), compute $\mathrm{MSE}(r)$ as above, and return
`(best_idx, best_mse)`:

- `best_idx`: integer array of shape `(rows, cols // group_size)`, the index into
  `clip_ratios` achieving the minimum MSE for that group.
- `best_mse`: `float64` array, same shape, the MSE value at `best_idx`.

## Example

```python
import numpy as np

# One group with 31 small values and one big outlier.
w = np.concatenate([np.full(31, 0.02), [0.3]])
clip_ratios = np.linspace(1.0, 0.5, 11)
best_idx, best_mse = awq_clip_search(w.reshape(1, -1), group_size=32, clip_ratios=clip_ratios)
# ratio 1.0 sets scale from the outlier -> crushes all 31 small values
# a smaller ratio clips the outlier but quantizes the other 31 far more precisely
# -> best_idx typically points to a ratio < 1.0
```

## What the gate checks

The gate builds a NumPy oracle that runs the identical grid sweep on a fixed test weight
matrix (roughly half the groups contain a single outlier element several times larger
than the rest of the group, so clipping should win in some groups but not others). It
checks:

- `idx_exact_match`: your `best_idx` array must exactly match the oracle's for every
  group (must be `1.0`).
- `mse_max_abs_err`: the max absolute error of your `best_mse` versus the oracle's, at
  most $10^{-6}$.
