## Context

A real "compound" compressed format combines structured sparsity with
quantization: 2:4 pruning zeros out 2 of every 4 consecutive weights, and
the 2 survivors in each block are then quantized with their own small
group scale. Because only survivors need to be stored, the natural scale
for a block is a statistic of the survivors *alone*:

$$
\text{scale} = \operatorname{mean}(|v| : v \text{ a survivor of the block}).
$$

The bug in this pipeline is a classic pruning/quantization integration
mistake: the code that computes the per-block scale still divides by the
**block width** (4) instead of the **survivor count** (2), silently
folding the two structurally-zeroed positions into the average as if they
were real values to be quantized. Since exactly half of every block's
slots are pruned zeros, this dilutes every block's scale by exactly 2x,
mis-calibrating the quantization codes for every surviving weight in the
tensor.

For a block $b = (w_0, w_1, w_2, w_3)$, quantized with `nbits` bits:

$$
q_{\max} = 2^{\text{nbits}-1} - 1, \qquad
\text{code}(v) = \operatorname{clip}\!\left(\operatorname{round}\!\left(\frac{v}{\text{scale}}\right), -q_{\max}, q_{\max}\right),
$$

$$
\hat{v} = \text{code}(v) \cdot \text{scale}.
$$

Pruned positions always dequantize to exactly $0$.

## Task

Fix `compound_prune_quantize_2_4(W, nbits=4)`.

`W` is a 2-D float array whose last dimension is a multiple of 4. For every
consecutive block of 4 elements along the last axis:

1. **2:4 prune**: zero the 2 smallest-magnitude elements of the block,
   keep the 2 largest (the survivors). If magnitudes tie, break ties by
   keeping the elements Python's stable `argsort` would rank as largest
(i.e. use `sorted` or list methods on the block's absolute values and keep the last
   two indices).
2. **Correct scale**: `scale = mean(|survivors|)`, computed from the
   surviving elements only — **not** from all 4 slots in the block. If a
   block has zero survivors (it was already all-zero), use `scale = 1.0`.
3. **Quantize/dequantize each survivor** with that scale as shown above.
   Pruned positions stay exactly `0.0`.

Return `W_hat`, a `float64` array with the same shape as `W`.

The bug in the starter divides the survivor magnitude sum by the fixed
block width `4` rather than the actual number of survivors, silently
treating the pruned zeros as if they contributed to the group's statistic.

## Example

```python

W = [[0.1, -0.2, 3.0, -4.0]]
W_hat = compound_prune_quantize_2_4(W, nbits=4)
# The two smallest-magnitude entries (0.1, -0.2) are pruned to 0.
# scale = mean(|3.0|, |-4.0|) = 3.5 -- computed from the 2 survivors only.
```

## What the gate checks

The gate rebuilds the correct pipeline (prune, then scale from survivors
only, then quantize) with an independent Python oracle across several
weight matrices and bit widths.

- `max_abs_err`: the maximum absolute error between your reconstructed
  `W_hat` and the oracle's must be at most `1e-9`.

Folding the pruned zeros into the scale's denominator halves every block's
scale, which changes essentially every quantization code in the tensor —
the buggy pipeline fails this gate by a wide margin.
