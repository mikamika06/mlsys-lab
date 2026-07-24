## Context

QLoRA's NF4 quantization splits a weight tensor into blocks of
`block_size` elements. Each block $b$ is normalized by its own absmax
constant $c_1^{(b)} = \max(|b|)$ and every value snapped to the 16-level
NF4 codebook. Storing one fp32 $c_1$ per block already costs
$32/\texttt{block\_size}$ bits per parameter on top of the 4-bit codes —
for `block_size=64` that is another $0.5$ bit/param, a meaningful
overhead relative to 4-bit weights.

**Double quantization** shrinks that overhead by quantizing the $c_1$
constants themselves. Every `outer_block` consecutive $c_1$ values are
grouped and quantized with an ordinary asymmetric (affine) min-max
8-bit quantizer, using $c_2 = (\max(c_1\text{-group}) -
\min(c_1\text{-group}))/255$ as the group's scale — needing only one
more fp32 per `outer_block` values of $c_1$ instead of per weight. The
bits-per-parameter overhead becomes

$$
\text{bits/param} = 4 + \frac{8}{\texttt{block\_size}} + \frac{32}{\texttt{block\_size} \cdot \texttt{outer\_block}} .
$$

For `block_size=64, outer_block=256` (QLoRA's defaults) this drops the
absmax overhead from $0.5$ bit/param to about $0.127$ bit/param.

## Task

Implement `nf4_double_quant_dequant`:

```python
def nf4_double_quant_dequant(weights: np.ndarray, block_size: int, outer_block: int):
    ...
```

* `weights` — array of any shape.
* `block_size` — level-1 NF4 block size.
* `outer_block` — number of consecutive $c_1$ values grouped for the
  level-2 8-bit quantization.

**Level 1**: flatten `weights`; for each block of `block_size` elements,
compute $c_1 = \max(|\text{block}|)$ (use $c_1 = 1$ for an all-zero
block), normalize the block by $c_1$, and snap every normalized value to
its nearest level in the 16-level NF4 codebook (the codebook is the
quantiles of a standard normal distribution at $(i+0.5)/16$ for $i = 0,
\dots, 15$).

**Level 2**: group every `outer_block` consecutive $c_1$ values and
quantize+dequantize each group with asymmetric affine min-max 8-bit
quantization (scale from that group's own min/max, zero-point rounded
from $-\min/\text{scale}$, codes clipped to $[0, 255]$), giving
dequantized $\hat c_1$.

Reconstruct every weight as $\hat w = \text{codebook}[\text{code}] \cdot
\hat c_1[\text{block index}]$, reshaped back to `weights`'s original
shape.

Return `(reconstructed_weights, bits_per_param)` where `bits_per_param`
is the formula above.

## Example

```python
import numpy as np

rng = np.random.default_rng(0)
w = rng.standard_normal(1000)

recon, bits = nf4_double_quant_dequant(w, block_size=64, outer_block=4)
# recon.shape == w.shape; bits == 4 + 8/64 + 32/(64*4) == 4.25
```

## What the gate checks

The gate, **max_abs_err**, compares your `reconstructed_weights` and
`bits_per_param` against an fp64 NumPy oracle across several shapes and
block-size combinations, including an all-zero block and a size not
evenly divisible by `block_size`. Both the reconstructed weights (which
must match the oracle's two-level quantize/dequantize path exactly, not
just be numerically "close" to the original weights) and the
bits-per-parameter number must match to within `1e-9`.
