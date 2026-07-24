## Context

Weight-only quantization has two independent knobs that both trade off
against reconstruction error: the **bit width** used per element, and the
**group size** — how many consecutive elements (along a row's input
dimension) share one `(scale, zero_point)` pair. Fewer bits per element
means a coarser grid; a larger group means one scale/zero pair has to fit
more values, so outliers within the group stretch the grid for everyone
sharing it.

For a weight matrix $W \in \mathbb{R}^{r \times c}$, row-wise **grouped
affine (asymmetric) quantization** with group size $g$ splits each row
into $c / g$ contiguous chunks. For a chunk of values $x$ with
$b$ bits:

$$
\text{scale} = \frac{\max(x) - \min(x)}{2^b - 1}, \qquad
\text{zero} = \mathrm{clip}\!\left(\mathrm{round}\!\left(\frac{-\min(x)}{\text{scale}}\right),\, 0,\, 2^b-1\right)
$$

$$
\text{code}_i = \mathrm{clip}\!\left(\mathrm{round}\!\left(\frac{x_i}{\text{scale}} + \text{zero}\right),\, 0,\, 2^b - 1\right), \qquad
\hat{x}_i = (\text{code}_i - \text{zero}) \cdot \text{scale}
$$

(if $\max(x) = \min(x)$ the chunk is constant, so $\hat{x} = x$ exactly).
"Per-tensor" quantization is the degenerate case of a single group spanning
the *entire* matrix (one shared scale/zero for everything). Shrinking the
group size (or growing the bit width) can only make the fit *at least as
good*: reconstruction MSE never increases as groups get finer, because a
finer group can always fall back to reproducing what a coarser group did
(same $\min$/$\max$ span or tighter).

## Task

Implement `bitwidth_group_mse_frontier`:

```python
def bitwidth_group_mse_frontier(
    W: np.ndarray,
    bit_options: list[int],
    group_size_options: list[int | None],
) -> np.ndarray:
    ...
```

- `W`: a 2-D `float64` array of shape `(rows, cols)`.
- `bit_options`: bit widths to sweep, e.g. `[2, 4]`.
- `group_size_options`: group sizes to sweep, each either a positive `int`
  that evenly divides `cols` (grouping is **per row**, along the columns),
  or `None` meaning **per-tensor** — a single group over the entire
  flattened matrix.

Return a 2-D array `mse` of shape `(len(bit_options), len(group_size_options))`
where `mse[i, j]` is the mean squared reconstruction error
$\frac{1}{rc}\sum (\hat{W} - W)^2$ of quantize-then-dequantize `W` with
`bit_options[i]` bits and group size `group_size_options[j]`, using the
formulas above.

## Example

```python
import numpy as np

W = np.array([[0.0, 1.0, 2.0, 100.0]])  # one row, one outlier
bitwidth_group_mse_frontier(W, [4], [None, 2])
# mse[0, 0] (per-tensor, one group of 4): the outlier stretches the scale
#   for the whole row, so the small values reconstruct poorly.
# mse[0, 1] (group_size=2, two groups of 2): the outlier's group absorbs
#   its own damage, and the [0.0, 1.0] group reconstructs almost exactly
#   -> strictly lower MSE than the per-tensor column.
```

## What the gate checks

The grader builds a fixed `(rows, cols)` weight matrix from a seeded NumPy
generator (Gaussian values plus a handful of injected outliers, so
different group sizes are genuinely distinguishable) and sweeps
`bit_options = [2, 4]` against
`group_size_options = [None, 128, 64, 32]` (cols is a multiple of all
three group sizes). It computes the reference `mse` array independently in
NumPy — same grouped affine quantize/dequantize formulas, applied
per-row-per-group and never calling your function or hardcoding numbers.

Two gates apply: `mse` is the relative error between your full `mse` array
and the oracle's (must be `<= 1e-6`), and `monotone` is `1.0` only if,
for every bit width, your own reported MSE is non-increasing as the group
size shrinks through the given `group_size_options` order (else `0.0`).
Ignoring `group_size` and always quantizing per-tensor, mixing up which
axis groups run along, or getting the scale/zero-point formula wrong will
fail the `mse` gate; reporting an MSE curve that isn't monotone (e.g. from
recomputing min/max over the wrong slice) fails the `monotone` gate even
if some individual cells happen to be close.
