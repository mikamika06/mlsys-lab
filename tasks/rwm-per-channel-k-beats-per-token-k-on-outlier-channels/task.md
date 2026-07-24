## Context

KIVI quantizes the key cache **per channel** rather than per token. To see
why, consider a symmetric uniform quantizer with $b$ bits applied to a group
of values $x$:

$$
s = \frac{\max(|x|)}{2^{\,b-1}-1}, \qquad
\hat x = \operatorname{clip}\!\big(\operatorname{round}(x/s),\, -(2^{b-1}-1),\, 2^{b-1}-1\big)\cdot s .
$$

For a key cache $K \in \mathbb{R}^{n\times d}$ (rows = tokens, columns =
RoPE-rotated channels), there are two natural ways to group elements that
share one scale $s$:

- **per-channel**: one scale per column, with $\max(|x|)$ taken over all $n$
  tokens in that channel.
- **per-token**: one scale per row, with $\max(|x|)$ taken over all $d$
  channels of that token.

Real key caches have a handful of **persistently high-variance channels**
(a few columns whose values are consistently much larger in magnitude than
the rest, across every token) while most channels stay small. Per-channel
quantization isolates each outlier channel into its own scale — it only
has to cover *that channel's* range, so every other channel keeps a tight,
low-error scale. Per-token quantization can't do this: every row contains
the same outlier channel, so every row's $\max(|x|)$ is dragged up by it,
and the coarse per-row step size crushes all the *other*, well-behaved
channels in that row too.

## Task

Implement `compare_k_quant_granularity(K, bits)`:

```python
def compare_k_quant_granularity(K: np.ndarray, bits: int):
    ...
```

- `K`: `(n_tokens, d_channels)` float64 key cache.
- `bits`: bit-width of the symmetric quantizer above.

Return `(mse_per_channel, mse_per_token)` as plain Python floats:

- `mse_per_channel` — mean squared reconstruction error of `K` quantized
  per-channel (one scale per column, `max(|x|)` over all rows).
- `mse_per_token` — mean squared reconstruction error of `K` quantized
  per-token (one scale per row, `max(|x|)` over all columns).

## Example

```python
K.shape  # (48, 16); a few columns have much larger magnitude than the rest
compare_k_quant_granularity(K, bits=4)
# -> (0.0004, 0.31)
# per-channel MSE is far below per-token MSE: the outlier channels each
# get their own scale, while per-token grouping forces every row's step
# size to cover the outlier's huge range.
```

## What the gate checks

The grader builds several seeded `K` matrices: most channels are small
Gaussian noise, and a handful of channels (different count/scale per case)
have a much larger standard deviation, at varying `(n_tokens, d_channels,
bits)`. For each case it computes the reference `(mse_per_channel,
mse_per_token)` with a NumPy oracle implementing the exact symmetric
quantize/dequantize formula above, once grouped by column and once by row.

The gate metric is `rel_err`: the relative L2 error between your returned
pair and the oracle's pair, worst case over all test cases, must be
`<= 1e-6`. In addition, any case where the oracle shows a clear gap
(`mse_per_token > 2 * mse_per_channel`) but your own returned numbers don't
have `mse_per_channel < mse_per_token` is treated as a hard failure — the
whole point of the task is reproducing that ordering, not just matching
numbers by coincidence. Swapping the two axes, using a single per-tensor
scale, or getting the quantizer formula wrong will violate the tolerance,
the ordering, or both.
