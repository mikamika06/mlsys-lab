## Context

"tinygemm" int4 weight-only quantization (used by PyTorch's `torch.ops.aten._weight_int4pack_mm`
family and similar CUDA kernels) packs each row of a weight matrix into 4-bit unsigned
codes, with a separate `(scale, zero_point)` pair per contiguous **group** of columns.
Unlike the symmetric schemes used for activations, this is an **asymmetric** grid:
the 16 representable codes are spread exactly across each group's own
`[min, max]` range, using a zero point expressed directly in real-value (float) units.

For a group of weights $w_1, \dots, w_g$ (here $g = 128$):

$$
\mathrm{min} = \min_i w_i, \qquad \mathrm{max} = \max_i w_i,
$$

$$
s = \frac{\mathrm{max} - \mathrm{min}}{15}, \qquad z = \mathrm{min}.
$$

($s$ is divided by $15$, not $16$, because a 4-bit code spans the $16$ integers
$0, 1, \dots, 15$ — the two endpoints of the range must land exactly on codes $0$ and
$15$.) Each weight is then quantized and dequantized as

$$
q = \mathrm{clip}\!\left(\mathrm{round}\!\left(\frac{w - z}{s}\right),\, 0,\, 15\right)
\in \{0, \dots, 15\},
\qquad
\hat{w} = q \cdot s + z .
$$

If a group is perfectly constant ($\mathrm{max} = \mathrm{min}$), use $s = 1$ instead of
$s = 0$ to avoid division by zero — every code in that group is then exactly $0$, and
$\hat{w} = z = w$ still reconstructs perfectly.

## Task

Implement `tinygemm_int4_quantize`:

```python
def tinygemm_int4_quantize(
    W: np.ndarray, group_size: int = 128
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ...
```

- `W`: `float64` array of shape `(rows, cols)`, where `cols` is an exact multiple of
  `group_size`.
- `group_size`: number of columns per quantization group (each row has
  `cols // group_size` independent groups).

Return `(codes, scale, zero_point, dequantized)`:

- `codes`: integer array of shape `(rows, cols)`, values in `[0, 15]`.
- `scale`: `float64` array of shape `(rows, cols // group_size)`.
- `zero_point`: `float64` array of shape `(rows, cols // group_size)`, in float-domain
  (real-value) units — i.e. `zero_point = min` of the group, not an integer code.
- `dequantized`: `float64` array of shape `(rows, cols)`, equal to
  `codes * scale + zero_point` broadcast per group.

## Example

```python
import numpy as np

W = np.array([[0.0, 1.0, 2.0, 3.0]])  # 1 row, group_size = 4
codes, scale, zp, deq = tinygemm_int4_quantize(W, group_size=4)
# scale = [[3/15]] = [[0.2]], zero_point = [[0.0]]
# codes = [[0, 5, 10, 15]]
# deq   = [[0.0, 1.0, 2.0, 3.0]]  (exact, since 4 levels evenly divide the range)
```

## What the gate checks

The gate builds a NumPy oracle that computes `min`/`max` per group, derives
`scale = (max - min) / 15` and `zero_point = min`, rounds and clips to `[0, 15]`, and
dequantizes. It checks, against a fixed test weight matrix (including a constant group
to exercise the `scale == 0` edge case):

- `codes_exact_match`: your `codes` array must exactly match the oracle's codes
  (must be `1.0`).
- `params_max_abs_err`: the max absolute error of your `scale` and `zero_point` versus
  the oracle's, must be at most $10^{-6}$.
- `max_abs_err`: the max absolute error of your `dequantized` reconstruction versus the
  oracle's, must be at most $10^{-5}$.
