## Context

KIVI-style KV-cache quantization treats the key cache and the value cache
**differently**, because their outlier structure is different: keys have a
few consistently-large channels (columns), so K is quantized **per-channel**
— each channel gets its own scale over a window of `group_size` tokens.
Values don't have that channel structure but do vary a lot token-to-token,
so V is quantized **per-token** — each token (row) gets its own scale over
a window of `group_size` channels. Using the wrong axis for either tensor
mixes unrelated magnitudes into one scale/zero-point and wrecks accuracy.

Both use the same asymmetric int-`bits` affine grid per group. For a group
of values $x$ (default `bits=4`, so $q_{max} = 2^4-1 = 15$):

$$
\text{scale} = \frac{\max(x) - \min(x)}{q_{max}}, \qquad
\text{zero} = \mathrm{clip}\!\left(\mathrm{round}\!\left(\frac{-\min(x)}{\text{scale}}\right),\, 0,\, q_{max}\right)
$$

$$
\text{code}_i = \mathrm{clip}\!\left(\mathrm{round}\!\left(\frac{x_i}{\text{scale}}\right) + \text{zero},\, 0,\, q_{max}\right), \qquad
\hat{x}_i = (\text{code}_i - \text{zero}) \cdot \text{scale}
$$

(if $\max(x) = \min(x)$ the group is constant and $\hat{x} = x$ exactly.)

## Task

Implement `quantize_dequantize_kv`:

```python
def quantize_dequantize_kv(K: list[list[float]], V: list[list[float]], group_size: int, bits: int=4) -> tuple[list[list[float]], list[list[float]]]:
    ...
```

- `K`, `V`: `(seq_len, channels)` `float64` arrays.
- `group_size`: positive `int`. Divides `seq_len` (for `K`'s grouping) and
  `channels` (for `V`'s grouping).
- `bits`: bit width, default 4.

Quantize-then-dequantize **K along axis 0** — split each *column* (channel)
into contiguous groups of `group_size` *rows* (tokens), each group with its
own scale/zero-point per the formulas above.

Quantize-then-dequantize **V along axis 1** — split each *row* (token) into
contiguous groups of `group_size` *columns* (channels), each group with its
own scale/zero-point.

Return `(K_hat, V_hat)`, same shapes as `K`, `V`.

## Example

```python

K = [[0.0, 100.0],
              [5.0, 101.0],
              [10.0, 102.0],
              [15.0, 103.0]]
V = [[0.0, 5.0, 100.0, 101.0]]

K_hat, V_hat = quantize_dequantize_kv(K, V, group_size=4)
# K: each COLUMN is one group of 4 tokens -> col0 uses scale=15/15=1.0
#    (exact grid), col1 uses scale=3/15=0.2 over [100,103] (near-exact).
# V: the single ROW is one group of 4 channels -> one scale spans the
#    whole 0..101 range, wasting most levels on the gap between clusters
#    (this is why V is grouped by CHANNEL count, not by all channels at
#    once, when `channels` is large -- group_size controls that window).
```

## What the gate checks

The grader builds three seeded `(seq_len, channels)` pairs of `K`/`V`
tensors (Gaussian with injected outliers) with different `group_size`
values, and computes the oracle reconstruction independently in Python:
`K` grouped along axis 0 (per-channel), `V` grouped along axis 1
(per-token), using the exact scale/zero/code/dequant formulas above —
never calling your function.

`rel_err` is the worst-case global relative L2 error between your
`(K_hat, V_hat)` and the oracle's, across all three cases (must be
`<= 1e-6` — this is a near bit-exact check on the per-group math, not a
compression-quality bound; using the wrong axis for either tensor, or
mixing up which one gets per-channel vs. per-token grouping, produces an
error orders of magnitude above this threshold). `max_abs_err` is the
same worst case measured as max elementwise absolute difference (also
`<= 1e-6`), catching a solution whose average error looks small but has
a few badly wrong elements (e.g. one mis-handled boundary group).
