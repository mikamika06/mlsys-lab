## Context

GGUF's Q4_K format is a *two-level* k-quant scheme built around a 256-value
super-block. Each super-block splits into 8 sub-blocks of 32 values, and
each sub-block gets its own asymmetric (scale + min) 4-bit quantizer — but
storing a full-precision scale and min for every 32 values would waste more
bits than the 4-bit codes themselves save. Q4_K instead quantizes the 8
sub-scales and 8 sub-mins *again*, as 6-bit codes relative to two
super-block-wide values $d$ (scale-of-scales) and $d_{\min}$
(scale-of-mins).

For sub-block $i$ inside a super-block, with $\mathrm{mn}_i =
\min(\text{sub}_i)$ and $\mathrm{mx}_i = \max(\text{sub}_i)$:

$$
ss_i = \frac{\mathrm{mx}_i - \mathrm{mn}_i}{63}, \qquad
mm_i = \frac{-\mathrm{mn}_i}{63}.
$$

The super-block scale and min are the largest of these eight values:

$$
d = \max_i ss_i, \qquad d_{\min} = \max_i mm_i.
$$

Each sub-scale and sub-min is then re-quantized to a 6-bit code relative to
its super-block value:

$$
sc_i = \operatorname{clip}\!\big(\operatorname{round}(ss_i / d \cdot 63),\, 0,\, 63\big), \qquad
mc_i = \operatorname{clip}\!\big(\operatorname{round}(mm_i / d_{\min} \cdot 63),\, 0,\, 63\big).
$$

Finally, every weight $w$ in sub-block $i$ gets a 4-bit code

$$
q = \operatorname{clip}\!\left(\operatorname{round}\!\left(\frac{w + d_{\min}\, mc_i}{d\, sc_i}\right),\, 0,\, 15\right),
$$

and dequantizes back as

$$
\hat{w} = d \cdot sc_i \cdot q \;-\; d_{\min} \cdot mc_i.
$$

## Task

Implement `q4k_quantize_superblock(x)` and
`q4k_dequantize_superblock(codes, sub_scales, sub_mins, d, dmin)`.

`x` is a `float32` array of shape `(rows, cols)`, with `cols` always a
multiple of 256.

`q4k_quantize_superblock` must:

1. Process each row in super-blocks of 256 values, each split into 8
   sub-blocks of 32 values, exactly as described above (treat a sub-block
   with `d == 0` as producing all-zero codes for that sub-block: `step ==
   0` means `q = 0`; likewise `sc = 0` when `d == 0` and `mc = 0` when
   `dmin == 0`).
2. Pack the 256 four-bit codes per super-block two per byte: within each
   16-value output byte-run for a sub-block, byte `k` holds value `2k` in
   its low nibble and value `2k+1` in its high nibble.

Return `(codes, sub_scales, sub_mins, d, dmin)`:

- `codes`: `uint8`, shape `(rows, cols // 2)`.
- `sub_scales`: `uint8`, shape `(rows, cols // 256, 8)` — the 6-bit `sc_i`
  codes, values in `[0, 63]`.
- `sub_mins`: `uint8`, shape `(rows, cols // 256, 8)` — the 6-bit `mc_i`
  codes, values in `[0, 63]`.
- `d`: `float16`, shape `(rows, cols // 256)`.
- `dmin`: `float16`, shape `(rows, cols // 256)`.

`q4k_dequantize_superblock` must reconstruct a `float32` array of shape
`(rows, cols)` using $\hat{w} = d \cdot sc_i \cdot q - d_{\min} \cdot mc_i$
for every value.

## Example

```python
import numpy as np

x = np.array([[0.0, 1.0, -2.0, 3.0] * 64], dtype=np.float32)  # one super-block

codes, sub_scales, sub_mins, d, dmin = q4k_quantize_superblock(x)
y = q4k_dequantize_superblock(codes, sub_scales, sub_mins, d, dmin)
# y.shape == x.shape; y approximates x
```

## What the gate checks

The gate builds its own NumPy Q4_K oracle following the exact scheme above
on several representative rows (smooth, linear-ramp, piecewise-constant,
random, and a two-super-block row).

- `codes_exact`: your packed 4-bit `codes`, and your `sub_scales` /
  `sub_mins` 6-bit codes, must exactly match the oracle's on every case
  (and every code must lie in its valid range).
- `rel_err`: the relative error of your dequantized reconstruction
  (round-tripped through your own `codes`/`sub_scales`/`sub_mins`/`d`/
  `dmin`) versus the oracle's reconstruction must be at most `1e-3`.

A solution that quantizes the sub-scales/sub-mins directly to 6 bits
without first computing them from `(max-min)/63` and `-min/63`, or that
mixes up which of `d`/`dmin` scales which array, will diverge from the
oracle's codes and fail `codes_exact`.
