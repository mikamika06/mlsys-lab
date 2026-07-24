## Context

In multi-head attention a projected tensor is split into $H$ independent heads,
each attended over separately, then merged back. Take a single sequence
$X \in \mathbb{R}^{T \times D}$ with $T$ tokens and hidden size $D = H\,d$, where
$H$ is the number of heads and $d = D / H$ the per-head size.

The **split** groups the last axis into heads and moves the head axis to the
front, so each head is a contiguous $T \times d$ matrix:

$$
X_{\text{split}} \in \mathbb{R}^{H \times T \times d},
\qquad
X_{\text{split}}[h, t, j] = X[t,\; h\,d + j].
$$

The **merge** is the exact inverse:

$$
X_{\text{merge}} \in \mathbb{R}^{T \times D},
\qquad
X_{\text{merge}}[t,\; h\,d + j] = X_{\text{split}}[h, t, j].
$$

Because the head axis is moved to the front, this is **not** a plain contiguous
reshape: element $[h,t,j]$ of the output does not sit at flat position
$h\,d + t\cdot\text{(something)} + j$ of the input. It is a reshape *composed
with a transpose*. The point of this exercise is to perform that rearrangement
with explicit index arithmetic so the mapping is fully in your hands.

## Task

Implement two functions:

```python
def split_heads(x: np.ndarray, num_heads: int) -> np.ndarray:
    ...

def merge_heads(heads: np.ndarray) -> np.ndarray:
    ...
```

- `split_heads` takes a 2-D array of shape `(T, D)` (with `D` divisible by
  `num_heads`) and returns an array of shape `(num_heads, T, D // num_heads)`
  satisfying `out[h, t, j] == x[t, h*head_dim + j]`.
- `merge_heads` takes an array of shape `(num_heads, T, head_dim)` and returns
  the `(T, num_heads * head_dim)` inverse satisfying
  `out[t, h*head_dim + j] == heads[h, t, j]`.

**You must move the data with explicit index arithmetic** — nested loops over
`(h, t, j)` and element assignment. You may allocate the output with
`np.empty`/`np.zeros`, but you may **not** use `reshape`, `transpose`,
`swapaxes`, `moveaxis`, or any other whole-array rearrangement primitive to do
the permutation. Those run in C and are rejected by the op-count gate (see
below). Results must be `float64`.

## Example

```python
import numpy as np
x = np.arange(12, dtype=float).reshape(2, 6)   # T=2, D=6
# x = [[ 0  1  2  3  4  5]
#      [ 6  7  8  9 10 11]]

heads = split_heads(x, 3)                       # H=3, head_dim=2 -> shape (3, 2, 2)
# heads[0] = [[0, 1], [6, 7]]      (columns 0:2)
# heads[1] = [[2, 3], [8, 9]]      (columns 2:4)
# heads[2] = [[4, 5], [10, 11]]    (columns 4:6)

back = merge_heads(heads)                        # shape (2, 6)
assert np.array_equal(back, x)
```

## What the gate checks

Two gates must both pass:

1. **`max_abs_err` $\le 10^{-6}$** — the grader builds the reference `split` and
   `merge` with NumPy and reports
   $\max_{h,t,j}\lvert \text{your}[h,t,j] - \text{reference}[h,t,j]\rvert$ over
   several fixed shapes (both directions are checked; `merge` is fed the oracle's
   correct split so it is tested independently).

2. **`op_count` $\ge 500$** — while your functions run, the grader counts
   Python line events with `sys.settrace`. Explicit per-element indexing emits
   roughly one event per element (thousands here); a `reshape`/`transpose`
   solution runs entirely in C and emits only a handful. A correct-but-vectorized
   answer therefore passes gate 1 but **fails** gate 2, which is exactly what
   forces the manual index arithmetic.
