## Context

QLoRA's **NF4** ("NormalFloat4") format is a real 4-bit codec used to store
frozen model weights. It is designed around a key fact: pretrained neural
network weights are, to a good approximation, zero-mean Gaussian. A *uniform*
4-bit codec (16 evenly-spaced levels between $-1$ and $1$) wastes most of its
levels on the tails of the distribution, where almost no weights live, and
under-resolves the dense cluster of values near zero. NF4 instead uses 16
**quantile levels of a standard normal distribution** — fixed constants,
shipped as a lookup table in production libraries (bitsandbytes) — so that
each level covers roughly equal *probability mass* rather than equal
*numeric range*.

A second, independent bug shows up in naive codecs: using a single **global**
`scale = max(abs(whole_tensor))` lets one outlier weight blow up the scale
for the entire tensor, crushing every other (much smaller) block of weights
towards zero. Production codecs instead compute the scale **per block**
(`block_size` contiguous elements), so each block's own dynamic range is used.

You are given a codec (`starter.py`) that gets *both* of these wrong: it uses
16 uniformly-spaced levels, and one global scale for the whole array. Your
job is to fix it into the real NF4-blockwise codec.

The fixed codec, for a 1-D weight vector $w$ split into contiguous blocks of
`block_size` elements:

For each block $b$:
$$
s_b = \max_i |w_i^{(b)}|\quad(\text{use } s_b = 1 \text{ if the block is all-zero})
$$
$$
\tilde{w}_i^{(b)} = w_i^{(b)} / s_b \qquad (\text{normalized into } [-1, 1])
$$
$$
c_i^{(b)} = \arg\min_{\ell \in L}\; \big|\tilde{w}_i^{(b)} - \ell\big|
$$
where $L$ is the fixed 16-value NF4 codebook:
```
L = [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
     -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
      0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
      0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0]
```
Reconstruction (dequantization): $\hat{w}_i^{(b)} = L[c_i^{(b)}] \cdot s_b$.

## Task

Fix `nf4_blockwise_dequant` so it implements the algorithm above:

```python
def nf4_blockwise_dequant(w: list[float], block_size: int=64) -> list[float]:
    ...
```

* `w` — 1-D `float64` list, `len(w)` divisible by `block_size`.
* Returns a `float64` array of the same shape: the quantize-then-dequantize
  round trip of `w` through the NF4-blockwise codec described above.
* Use the exact 16-value codebook `L` given above (this is the real NF4
  codebook used by bitsandbytes/QLoRA).
* Use vectorised Python; no explicit Python loops over elements.

## Example

```python
w = [0.001, -0.02, 0.0005, 0.03, -0.001, 0.0002, 0.0, 0.025]
w_hat = nf4_blockwise_dequant(w, block_size=8)
# w_hat should stay close to w -- e.g. |w_hat - w| is small relative to
# max(abs(w)) == 0.03, because the single block's own absmax (0.03) is used
# as the scale, and the quantile levels concentrate resolution near zero.
```

## What the gate checks

The grader builds a realistic weight vector (zero-mean Gaussian, fixed seed)
and compares your reconstruction against two things:

* **rel_err** — relative L2 error between your reconstruction and an
  independent Python oracle that implements the exact algorithm above
  (blockwise absmax + nearest-NF4-level lookup). Tight tolerance: any correct
  implementation of the same deterministic table lookup should match almost
  exactly.
* **beats_naive** — your reconstruction's MSE against the true weights must
  be strictly lower than the MSE of the broken codec in `starter.py` (uniform
  16-level codebook + one global scale) on the same data. This is computed
  from real numbers each run, not hardcoded — it is the actual "NF4-blockwise
  beats uniform-global-int4 on Gaussian weights" effect this task is about.
