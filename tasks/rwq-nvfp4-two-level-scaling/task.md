## Context

NVFP4 is NVIDIA's 4-bit floating-point weight format. A single, coarse per-tensor scale
is not enough to keep 4-bit elements accurate, and a scale per tiny block wastes memory
if stored at full precision — NVFP4 splits the difference with **two-level scaling**:

1. A single **per-tensor** `float32` scale, derived from the whole tensor's amax so
   that per-block scales (computed next) land inside the range an 8-bit float can hold:

$$
s_{\text{tensor}} = \frac{\max_i |w_i|}{6 \cdot 448},
$$

   where $6$ is the max magnitude of an E2M1 (4-bit float) element and $448$ is the max
   magnitude of an E4M3 (8-bit float) value.

2. A **per-16-element block** scale, computed from each block's own amax so that the
   block's elements land inside $[-6, 6]$, then folded through $s_{\text{tensor}}$ and
   rounded to the nearest representable **E4M3** value (its native storage format):

$$
s_b = \mathrm{snap}_{\text{E4M3}}\!\left(\frac{\max_i |w_{b,i}| / 6}{s_{\text{tensor}}}\right).
$$

   The *effective* scale actually used for block $b$ is $s_{\text{tensor}} \cdot s_b$.

3. Each element is normalized by its block's effective scale and rounded to the nearest
   representable **E2M1** value (its native storage format) — the 8 non-negative E2M1
   magnitudes are $\{0, 0.5, 1, 1.5, 2, 3, 4, 6\}$, each carrying the original sign:

$$
q_{b,i} = \mathrm{snap}_{\text{E2M1}}\!\left(\frac{w_{b,i}}{s_{\text{tensor}} \cdot s_b}\right),
\qquad
\hat{w}_{b,i} = q_{b,i} \cdot s_{\text{tensor}} \cdot s_b .
$$

If a block's amax is exactly $0$, its E4M3 scale is $0$ and its elements are all $0$
(no division needed).

## Task

Implement `nvfp4_two_level_quantize`:

```python
def nvfp4_two_level_quantize(w: list[float], block_size: int=16) -> tuple[float, list[float], list[float], list[float]]:
    ...
```

- `w`: `float64` array of shape `(n,)`, `n` an exact multiple of `block_size`.
- `block_size`: number of elements per E4M3 block scale (always 16 in this task).

Return `(global_scale, block_scales_e4m3, codes, dequantized)`:

- `global_scale`: Python float, $s_{\text{tensor}} = \max|w| / (6 \cdot 448)$.
- `block_scales_e4m3`: `float64` array of shape `(n // block_size,)`, each entry snapped
  to the nearest non-negative E4M3-representable value.
- `codes`: `float64` array, same shape as `w`, each entry one of the 16 signed E2M1
  values.
- `dequantized`: `float64` array, same shape as `w`, equal to
  `codes * (global_scale * block_scales_e4m3)` broadcast per block.

A finite E4M3 value with exponent field $e \in \{0,\dots,15\}$ (excluding the NaN
pattern $e=15$, mantissa $=7$) and 3-bit mantissa $m$ has magnitude
$2^{-6} \cdot (m/8)$ when $e=0$ (subnormal) or $2^{e-7} \cdot (1 + m/8)$ when $e \ge 1$
(normal); "snap to nearest E4M3" means nearest value in that finite set, clipped to
$[0, 448]$.

## Example

```python

w = [3.0] * 16  # one block, all elements equal
gs, bs, codes, deq = nvfp4_two_level_quantize(w, block_size=16)
# tensor amax = 3.0  -> global_scale = 3.0 / (6*448)
# block amax = 3.0   -> block_scale_fp32 = 3.0/6 = 0.5
#   0.5 / global_scale = 448.0, already exactly E4M3-representable -> block_scales_e4m3 = 448.0
# effective scale = global_scale * 448.0 = 3.0/6 = 0.5
# normalized = 3.0 / 0.5 = 6.0 -> nearest E2M1 magnitude is 6.0 exactly -> codes = 6.0
# dequant = 6.0 * 0.5 = 3.0  (lossless here, since amax hits an exact grid point)
```

## What the gate checks

The gate builds a Python oracle running the identical two-level scaling pipeline on a
fixed test weight vector (block amax values spread around a common baseline, plus one
all-zero block). It compares, against the oracle:

- `global_scale_err`: absolute error of your `global_scale`, must be at most $10^{-6}$.
- `block_scale_max_abs_err`: max absolute error of your `block_scales_e4m3`, at most
  $10^{-6}$.
- `codes_max_abs_err`: max absolute error of your `codes`, at most $10^{-6}$.
- `max_abs_err`: max absolute error of your `dequantized` reconstruction, at most
  $10^{-6}$.
