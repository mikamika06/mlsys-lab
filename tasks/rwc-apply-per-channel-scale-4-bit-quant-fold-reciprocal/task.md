## Context

Low-bit weight quantization hurts most where a weight matrix has a few
**input channels with much larger magnitude** than the rest: a group-wise
quantizer's scale is set by the group's own min/max, so one big-magnitude
channel forces a coarse step size that then crushes every small,
"normal" channel sharing that group.

AWQ's fix: before quantizing, multiply each input channel $j$ by a
positive **smoothing factor** $s_j$ chosen so that channels are closer in
magnitude to each other (channels that were too large get divided back
down implicitly by choosing per-channel scales, or more precisely — the
transform below always changes *which* values get rounded, without
changing the math in exact arithmetic):

$$
W' = W \, \mathrm{diag}(s), \qquad \hat{W} = Q_{\text{4-bit}}(W'), \qquad
\text{output} = X \, \mathrm{diag}(s)^{-1} \, \hat{W}^\top
$$

Because $\mathrm{diag}(s)$ and $\mathrm{diag}(s)^{-1}$ are exact inverses,
if $Q$ were the identity this would compute exactly $X W^\top$ — the
scale is invisible to the *math*, only to the *rounding*. The whole point
of AWQ is to pick $s$ so the group-wise quantizer $Q$ rounds more
accurately; this task only applies an **already-chosen**, fixed $s$.

$Q_{\text{4-bit}}$ is a uniform affine (asymmetric) min-max quantizer
applied **per group of `group_size` consecutive input channels, per
output row** — i.e. for each output row and each block of `group_size`
columns, its own scale $s_g = (\max - \min)/(2^{\text{bits}}-1)$ and
zero-point, applied only within that block:

$$
\hat{x} = \left(\operatorname{clip}\!\left(\operatorname{round}\!\left(\frac{x}{s_g} + z_g\right),\, 0,\, 2^{\text{bits}}-1\right) - z_g\right) \cdot s_g
$$

## Task

Implement `awq_apply_fixed_scale(W, s, X, group_size, bits=4)`:

```python
def awq_apply_fixed_scale(W: np.ndarray, s: np.ndarray, X: np.ndarray, group_size: int, bits: int = 4) -> np.ndarray:
    ...
```

- `W`: `(out_features, in_features)` weight matrix.
- `s`: `(in_features,)` positive per-channel smoothing scale.
- `X`: `(batch, in_features)` activations.
- `group_size`: quantizer group size along the input-channel axis
  (`in_features` is always a multiple of `group_size`).
- `bits`: quantizer bit-width.

Follow the four steps above exactly (scale weight columns by `s`,
group-wise fake-quantize, fold `1/s` into `X`, matmul) and return
`output`, shape `(batch, out_features)`.

## Example

```python
W.shape, s.shape, X.shape  # (6, 8), (8,), (5, 8)
awq_apply_fixed_scale(W, s, X, group_size=4, bits=4).shape
# -> (5, 6)
```

## What the gate checks

The gate runs several `(W, s, X, group_size, bits)` combinations from
seeded generators, including one where `s` is all ones (so the transform
should reduce to plain grouped fake-quantization of `W` with no folding
effect at all). For every case the reference computes `output` with the
exact four-step pipeline above in NumPy. Your output is compared to it
with `max_abs_err < 1e-9` — since both the reference and a correct
solution run the *identical* deterministic pipeline (not an
approximation of the unquantized layer), a correct implementation
matches almost exactly. A solution that quantizes `W` group-wise **before**
applying `s` (i.e. computes `Q(W) * s` instead of `Q(W * s)`) changes
which values the rounding operates on and will disagree with the oracle
on every case with a non-trivial `s`.
