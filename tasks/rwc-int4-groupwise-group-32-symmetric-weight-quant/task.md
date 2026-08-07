## Context

"8da4w" weight-side quantization (used by torchao and similar
weight-only int4 schemes) quantizes a weight matrix **symmetrically**
(zero-point pinned to 0) and **group-wise**: rather than one scale per
row, each row is split into contiguous chunks of `group_size` values
(here $32$), and each chunk gets its own independent scale, computed from
that chunk's own maximum absolute value:

$$
a = \max_j |w_j|, \qquad s = \frac{a}{8}, \qquad
c_j = \operatorname{clip}\!\left(\operatorname{round}\!\left(\frac{w_j}{s}\right),\, -8,\, 7\right)
$$

Dividing by $8$ (not $7$) uses the full signed 4-bit range $[-8, 7]$: the
largest-magnitude value in the group always rounds to exactly $\pm 8$ or
close to it, landing at (or very near) the extreme code, while $-8$
itself remains reachable even though only $+7$ is the positive extreme
(a standard asymmetric-range/symmetric-scale int4 convention). Smaller
groups mean scales track local magnitude more tightly, at the cost of
storing more scale values.

## Task

Implement `int4_groupwise_quant(W, group_size=32)`:

```python
def int4_groupwise_quant(W: list[list[float]], group_size: int=32) -> tuple[list[list[int]], list[list[float]]]:
    ...
```

- `W`: `(rows, cols)` weight matrix; `cols` is always a multiple of
  `group_size`.

For every row, split its columns into contiguous, non-overlapping groups
of `group_size` values. For each group independently, compute
`amax = max(abs(group))`, then `scale = amax / 8` (use `scale = 1.0`
instead when `amax == 0`, to avoid dividing by zero), then
`code = clip(round(group / scale), -8, 7)`.

Return `(codes, scales)`:

- `codes`: `(rows, cols)` integer array of int4 codes, every value in
  `[-8, 7]`.
- `scales`: `(rows, cols // group_size)` float array, one scale per row
  per group.

## Example

```python
W.shape  # (8, 64), group_size=32 -> 2 groups per row
codes, scales = int4_groupwise_quant(W, group_size=32)
codes.shape   # (8, 64)
scales.shape  # (8, 2)
```

## What the gate checks

The gate builds several weight matrices from seeded generators, with
per-group magnitude deliberately varied so different groups in the same
row land on very different scales (checking that grouping is genuinely
independent per chunk, not a single row-wide scale), plus an explicit
all-zero group to check the divide-by-zero fallback. For every case the
reference computes `codes`/`scales` with Python exactly as described.

Your `codes` must match the oracle's **exactly** (integer equality, and
every value must lie in `[-8, 7]`), your `scales` must match within a
tight tolerance, and the dequantized reconstruction `codes * scales`
(broadcast per group) must also match the oracle's reconstruction
closely. A solution that computes one scale per **row** (using the whole
row's max instead of each group's own max) will produce plausible-looking
codes but disagree with the oracle as soon as a row has groups with
different characteristic magnitudes.
