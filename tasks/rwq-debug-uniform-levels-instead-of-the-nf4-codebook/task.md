## Context

NF4 (NormalFloat4) is the 4-bit codebook QLoRA uses to quantize a
frozen base model's weights. It is **not** a uniform grid — its 16
levels are quantiles of a standard normal distribution, packed densely
near zero (where most weight mass sits) and sparsely near the extremes:

$$
\texttt{NF4} = [-1.0,\ -0.6962,\ -0.5251,\ -0.3949,\ -0.2844,\ -0.1848,\ -0.0911,\ 0.0,
$$
$$
0.0796,\ 0.1609,\ 0.2461,\ 0.3379,\ 0.4407,\ 0.5626,\ 0.7230,\ 1.0]
$$

Before snapping to this codebook, weights are **block-normalized**:
the tensor is split into contiguous blocks (blocksize 64 is the
standard bitsandbytes default), and each block gets its **own**
absmax scale — not one scale for the whole tensor — so a single
outlier block doesn't wash out the resolution everywhere else. For a
block $b$ of values:

$$
\text{scale}_b = \max_i |b_i| \quad (\text{or } 1 \text{ if all-zero}), \qquad
\hat{b}_i = \frac{b_i}{\text{scale}_b}
$$

$$
\text{code}_i = \operatorname*{arg\,min}_{k \in \{0,\dots,15\}} \left| \hat{b}_i - \texttt{NF4}[k] \right|
$$

## Task

Fix `nf4_quantize_indices`:

```python
def nf4_quantize_indices(w: list[float], block_size: int=64) -> list[int]:
    ...
```

- `w`: 1-D `float64` array, length a multiple of `block_size`.
- `block_size`: elements per normalization block (default 64).

The supplied version has **two** bugs: it uses 16 evenly spaced levels
(`list(linspace(-1, 1, 16))`) instead of the real NF4 codebook above, and
it normalizes by one global absmax over the whole array instead of a
separate absmax **per block**. Fix both: split `w` into contiguous
blocks of `block_size`, normalize each block by its own absmax
(1.0 if the block is all-zero), and for every normalized value return
the index (0..15) of the nearest NF4 codebook level.

Return a 1-D `int64` array of indices, same length as `w`.

## Example

```python

w = [0.02, -0.01, 0.0, 0.019] + [0.0]*60 # one block of 64
nf4_quantize_indices(w, block_size=64)
# block absmax = 0.02 -> normalized = [1.0, -0.5, 0.0, 0.95, 0,0,...]
# 1.0 snaps to NF4[15]=1.0, -0.5 snaps to NF4[2]=-0.5251 (closest),
# 0.0 snaps exactly to NF4[7]=0.0, 0.95 snaps to NF4[15]=1.0 (closest)
```

## What the gate checks

The grader builds several seeded `float64` arrays (Gaussian, scaled
small, length a multiple of 64) and computes the reference index array
independently in Python — per-block absmax normalization against the
true NF4 codebook, argmin over the 16 levels — never calling your
function.

`index_match_frac` is the fraction of returned indices, across every
element of every case, that exactly equal the oracle's (must be
`>= 1.0`, i.e. every single index correct — these are discrete codes,
there's no "close enough"). Using evenly spaced levels instead of the
true NF4 quantiles mis-assigns most indices except the very extremes
and the middle-ish crossover points; using a single global scale
instead of per-block absmax additionally shifts every block whose
local magnitude differs from the global one.
