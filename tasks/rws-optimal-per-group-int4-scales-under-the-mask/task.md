## Context

A pruned-then-quantized weight $W\odot M$ (mask $M\in\{0,1\}$ zeroing
pruned entries) is grouped along `in_features` and int4-quantized per
group. The naive choice of each group's scale is `max(|group|)/qmax`
— but what actually matters for accuracy is not the raw weight
reconstruction error, it's the **output** error once the weight is
multiplied by real calibration activations $X$:

$$
\mathcal{L} = \frac{1}{n\cdot O}\left\lVert X(W\odot M)^\top - X\hat W^\top\right\rVert_F^2
$$

Because $Y = X(W\odot M)^\top$ decomposes independently **per output
row** ($Y_{:,o} = X\cdot(W\odot M)_{o,:}^\top$), the search below
optimizes row by row. Within a row, groups interact through $X$
(their errors add inside the same squared norm), so this task uses one
left-to-right **greedy coordinate-descent sweep**: initialize every
group with the naive scale, then revisit each group once, grid-search
a scale multiplier $\alpha$ around the naive scale, and always keep
whichever $\alpha$ minimizes the row's *current total* output error
(with every other group's dequantized value held at whatever it
currently is).

## Task

Implement `optimal_group_scales_under_mask`:

```python
def optimal_group_scales_under_mask(W: list[list[float]], M: list[list[float]], X: list[list[float]], group_size: int, bits: int=4, alphas: list[float]=None):
    ...
```

- `W`: `(O, I)` `float64` weight matrix.
- `M`: `(O, I)` mask (`0`/`1`), same shape.
- `X`: `(n, I)` `float64` calibration activations.
- `group_size`: contiguous groups along axis 1 (`I % group_size == 0`).
- `bits`: quantizer bit width, $q_{\max}=2^{\text{bits}-1}-1$.
- `alphas`: 1-D array of scale multipliers to grid-search; if `None`,
use `[0.6 + i * 0.1 for i in range(9)]` (note $\alpha=1$ — the naive scale —
  is always one of the candidates).

Let `Wm = W * M`.

1. **Initialize** `what` (the working dequantized approximation of
   `Wm`, same shape) with the *naive* per-group quantizer applied to
   every group of every row: `scale = max(|seg|)/qmax` (or `1.0` if
   the group is all-zero), `dequant = scale * clip(round(seg/scale), -qmax, qmax)`.
2. For each row `o`, **one left-to-right pass** over its groups: for
   group `g`, try every `alpha` in `alphas`, computing
   `scale = alpha * max(|seg|) / qmax` (or `1.0` if all-zero) and the
   resulting `dequant` for *that group only*; plug it into `what[o]`
   (every other group of that row stays at its current value) and
   measure `err = sum((X @ (Wm[o] - what[o]))**2)`. Commit whichever
   `alpha` gives the lowest `err` (`scale`, `dequant`) into `what[o]`
   before moving to the next group.
3. After processing every row, compute
   `mse = mean((X @ Wm.T - X @ what.T)**2)` (over all `n*O` entries).

Return `(group_scales, mse)`:
- `group_scales`: `(O, I // group_size)` `float64`, the committed
  scale for every row/group (from step 2).
- `mse`: `float`, from step 3.

## Example

```python
import random; rng = random.Random(0)
W = rng.standard_normal((4, 16))
M = [[1.0 for _ in range(len(W[0]))] for _ in range(len(W))]
X = rng.standard_normal((10, 16))
scales, mse = optimal_group_scales_under_mask(W, M, X, group_size=8)
# scales.shape == (4, 2); mse is the achieved X-weighted output MSE,
# no worse than the naive (alpha=1 for every group) baseline.
```

## What the gate checks

The grader builds several seeded `(W, M, X, group_size)` cases and runs
the exact algorithm above independently in Python (same init, same
one-pass greedy sweep, same `alphas` grid) to get a reference `mse`, and
separately computes the **naive** baseline `mse` (every group at
`alpha=1`, no search) for comparison.

`mse_rel_err` is the relative error between your returned `mse` and the
oracle's optimized `mse`, across all cases (must be `<= 1e-6`) — this
is a deterministic search over a fixed grid, so any real deviation
(wrong candidate set, wrong row/group loop order, or evaluating the
wrong slice) produces a clearly different final `mse`.
`mse_vs_naive_margin` is `naive_mse - your_mse`, worst case across
cases (must be `>= -1e-9`, i.e. never worse than naive) — because
`alpha=1` is always in the candidate grid, a correct greedy search can
never end up worse than the naive baseline; returning it anyway (e.g.
by not actually running the search, or searching but not committing
improvements) fails this even if `mse_rel_err` happens to look
plausible.
