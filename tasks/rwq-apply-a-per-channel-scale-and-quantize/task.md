## Context

AWQ (Activation-aware Weight Quantization) observes that a small
fraction of weight *channels* — the ones multiplied by large-magnitude
activations — dominate a layer's quantization error, while the rest
tolerate a coarse int4 grid fine. Instead of protecting those channels
by keeping them in higher precision (which fragments the kernel), AWQ
**migrates a scale** into the weight before quantizing: multiply the
salient input channels of $W$ up (spreading them across more of the
int4 grid, so rounding hurts less) and divide the matching channels of
$X$ down by the same factor, so the layer's output is mathematically
unchanged. For a per-input-channel scale vector $s > 0$:

$$
W' = W \odot s, \qquad X' = X \oslash s \qquad \implies \qquad X' W'^\top = X W^\top
$$

(scale one side up, the other down by the same factor per channel — the
product is invariant). *Then* $W'$ — now with its salient channels
easier to represent — is quantized with a standard int-`bits`
group-affine grid, one scale/zero-point per contiguous group of
`group_size` input channels, **per output row**:

$$
\text{scale} = \frac{\max(g) - \min(g)}{2^{\text{bits}}-1}, \qquad
\text{zero} = \mathrm{clip}\!\left(\mathrm{round}\!\left(\frac{-\min(g)}{\text{scale}}\right),\, 0,\, 2^{\text{bits}}-1\right)
$$

$$
\text{code}_i = \mathrm{clip}\!\left(\mathrm{round}\!\left(\frac{g_i}{\text{scale}}\right) + \text{zero},\, 0,\, 2^{\text{bits}}-1\right), \qquad
\hat{g}_i = (\text{code}_i - \text{zero}) \cdot \text{scale}
$$

## Task

Implement `awq_scale_and_quantize`:

```python
def awq_scale_and_quantize(W: np.ndarray, X: np.ndarray, s: np.ndarray, group_size: int, bits: int = 4):
    ...
```

- `W`: `(out_features, in_features)` `float64` — a Linear layer's weight.
- `X`: `(batch, in_features)` `float64` — activations feeding it.
- `s`: `(in_features,)` `float64`, positive per-input-channel scale.
- `group_size`: positive `int` dividing `in_features`.
- `bits`: bit width for the weight quantizer, default 4.

1. Compute `W' = W * s` (broadcast per input channel / column) and
   `X' = X / s` (same per-channel scale, inverted).
2. Quantize `W'` per output row (`axis=1`, i.e. along `in_features`),
   in contiguous groups of `group_size`, with the int-`bits` group-affine
   formulas above, to get `W_hat` (same shape as `W'`).

Return `(Y_identity, Y_quant)`:
- `Y_identity = X' @ W'.T` — should equal `X @ W.T`.
- `Y_quant = X' @ W_hat.T` — the actual quantized-layer output.

## Example

```python
import numpy as np

W = np.array([[1.0, 100.0], [2.0, 50.0]])   # out=2, in=2
X = np.array([[1.0, 1.0]])                   # batch=1
s = np.array([1.0, 0.1])                     # shrink the huge channel

Wp = W * s          # [[1, 10], [2, 5]]      -- channel 1 now on a sane scale
Xp = X / s          # [[1, 10]]
Y_identity = Xp @ Wp.T   # equals X @ W.T == [[101, 52]]
# int4 group quant of Wp is now far more accurate than quantizing the
# original W directly, where the "100" and "50" entries would dominate
# every group's scale and crush the small "1"/"2" entries to ~0.
```

## What the gate checks

The grader builds several seeded `(W, X, s, group_size)` cases (Gaussian
`W`/`X`, `s` drawn uniformly from `[0.5, 3.0]`) and computes both the
direct `X @ W.T` and the group-quantized `X' @ W_hat.T` independently in
NumPy, never calling your function.

`identity_max_abs_err` is the worst-case max elementwise absolute
difference between your `Y_identity` and the oracle's direct `X @ W.T`
across all cases (must be `<= 1e-6`) — this is a pure floating-point
identity (no quantization involved), so it isolates a bug in how you
apply/invert the scale from a bug in the quantizer itself.
`quant_max_abs_err` is the worst-case max elementwise absolute
difference between your `Y_quant` and the oracle's quantized output
(must be `<= 1e-5`) — this catches a wrong int-`bits` group formula or
wrong grouping axis even when the scale-migration identity above is
correct.
