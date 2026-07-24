## Context

A per-tensor activation quantizer has to pick one scale for the whole
tensor, and that scale is set by the single largest-magnitude channel. In
real transformer activations a handful of channels routinely have magnitude
10-100x larger than the rest, so that shared scale crushes every ordinary
channel down to a couple of quantization levels. This channel imbalance
*is* the "quantization difficulty" of an activation tensor.

For an activation matrix $X \in \mathbb{R}^{T \times C}$ (T tokens, C
channels), define the per-channel peak magnitude and the cross-channel
"peakiness" ratio

$$
a_j = \max_t |X_{t,j}|, \qquad
r_j = \frac{a_j}{\operatorname{mean}_k(a_k)}.
$$

$r_j \gg 1$ means channel $j$ is a magnitude outlier relative to the
average channel — exactly the kind of channel that dictates (and wastes)
the shared per-tensor scale.

SmoothQuant migrates this difficulty from activations into weights with a
per-channel scale

$$
s_j = \frac{a_j^{\alpha}}{\big(\max_i |W_{i,j}|\big)^{1-\alpha}}, \qquad \alpha \in [0, 1],
$$

then rescales $X \to X \cdot \operatorname{diag}(s)^{-1}$ (compensated by
folding $\operatorname{diag}(s)$ into $W$, so the linear layer's output is
unchanged). Because $s_j$ grows with $a_j$, the channels that were the
worst outliers get shrunk the most, and the *cross-channel* peakiness ratio
$r_j$ becomes far more uniform.

## Task

Implement `channel_peakiness_before_after(X, W, alpha=0.5)`.

- `X`: activation samples, shape `(n_tokens, C)`.
- `W`: weight matrix, shape `(n_out, C)`, with column `j` aligned to
  activation channel `j`.
- `alpha`: SmoothQuant migration strength in `[0, 1]`.

Steps:

1. `amax_X[j] = max_t |X[t, j]|`, `amax_W[j] = max_i |W[i, j]|`.
2. `ratio_before[j] = amax_X[j] / mean(amax_X)`.
3. `s[j] = amax_X[j] ** alpha / amax_W[j] ** (1 - alpha)`.
4. `amax_X_smoothed[j] = amax_X[j] / s[j]` (this equals the per-channel max
   of the migrated activations `X / s`, since scaling by a positive
   constant commutes with `max(|·|)`).
5. `ratio_after[j] = amax_X_smoothed[j] / mean(amax_X_smoothed)`.

Return the tuple `(ratio_before, ratio_after)`, both `float64` NumPy arrays
of shape `(C,)`.

## Example

```python
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(size=(64, 8))
X[:, 3] *= 40.0        # channel 3 is a big outlier
W = rng.normal(size=(4, 8)) * 0.1

ratio_before, ratio_after = channel_peakiness_before_after(X, W, alpha=0.5)
# ratio_before[3] is far above 1; ratio_after[3] should be noticeably smaller
assert ratio_after.max() < ratio_before.max()
```

## What the gate checks

The gate rebuilds the same computation with an independent NumPy oracle
across several outlier-channel activation tensors (different shapes,
outlier counts, outlier magnitudes, and `alpha` values):

- `rel_err`: the relative L2 error between your two returned ratio vectors
  (concatenated) and the oracle's must be at most `1e-6`.
- `peak_drop_ok`: for every test case, the maximum of your `ratio_after`
  must be strictly smaller than the maximum of your `ratio_before` — i.e.
  migration must actually reduce the worst channel's peakiness, not just
  produce numbers that happen to be close to the oracle's.

A solution that computes `ratio_after` from the *un*-migrated activations
(forgetting to divide by `s`) will return `ratio_after == ratio_before` and
fail `peak_drop_ok` even though each individual number still "looks like" a
valid ratio.
