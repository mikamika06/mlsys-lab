## Context

FP8 **W8A8** inference quantizes both the weights *and* the activations to
8‑bit floating point (E4M3: 1 sign bit, 4 exponent bits, 3 mantissa bits,
exponent bias 7, max finite magnitude $448$) before the matmul, then
dequantizes the result. This is the scheme used by production FP8 GEMM
kernels (e.g. vLLM / TensorRT‑LLM "dynamic" FP8 paths):

- **Weights** get a single **per-tensor** scale, since a weight matrix is
  static and its whole-tensor $\max|\cdot|$ is known ahead of time.
- **Activations** get a **per-token** scale — one scale *per column* of the
  activation matrix — computed on the fly from the current batch (no
  calibration pass, hence "dynamic"). Per-token scaling matters because
  different tokens can have very different activation magnitudes; a single
  tensor-wide scale would waste dynamic range on outlier tokens and crush
  the rest.

Given weights $W \in \mathbb{R}^{M\times K}$ and activations
$X \in \mathbb{R}^{K\times N}$ (columns = tokens), the pipeline is:

$$
s_w = \frac{\max|W|}{448}, \qquad
s_x[j] = \frac{\max_k |X_{kj}|}{448} \quad (j = 1,\dots,N)
$$

$$
\hat W = \mathrm{cast}_{\text{E4M3}}\!\left(\frac{W}{s_w}\right), \qquad
\hat X_{:,j} = \mathrm{cast}_{\text{E4M3}}\!\left(\frac{X_{:,j}}{s_x[j]}\right)
$$

$$
Y = \big(\hat W \, \hat X\big) \odot \big(s_w \cdot s_x\big)
$$

where the final elementwise product broadcasts $s_w$ (a scalar) and $s_x$
(a length-$N$ vector, one value per output column/token) over the raw
matmul of the quantized operands, dequantizing it back to real units.

The E4M3 grid is not evenly spaced: subnormal values are
$\pm 2^{-6}\cdot\frac{m}{8}$ (mantissa $m \in \{0,\dots,7\}$) and normal
values are $\pm 2^{e-7}\cdot\left(1+\frac{m}{8}\right)$ for exponent
$e \in \{1,\dots,15\}\setminus\{15\}$ combined with mantissa $7$ (which is
reserved for NaN). Casting a real number to E4M3 means snapping it to the
*nearest* value on that grid, after clamping to $\pm 448$.

## Task

Implement `fp8_dynamic_matmul`:

```python
def fp8_dynamic_matmul(W: list[list[float]], X: list[list[float]]) -> list[list[float]]:
    ...
```

* `W` — 2‑D array of shape $(M, K)$, the weight matrix.
* `X` — 2‑D array of shape $(K, N)$, the activation matrix ($N$ tokens).

Compute the per-tensor scale for `W` and the per-token (per-column) scale
for `X` exactly as above (guard against an all-zero tensor/column by using
scale $1.0$ in that case, to avoid a division by zero). Cast both scaled
operands to the nearest representable E4M3 value, matmul the quantized
operands, then dequantize by multiplying back by `scale_w * scale_x[token]`
(broadcast over columns). Return `Y` with shape $(M, N)$.

Use vectorised Python only — build the 256-point E4M3 value grid once and
snap each element to its nearest grid point (e.g. via binary search),
rather than looping in Python.

## Example

```python
W = [[1.0, -2.0], [0.5, 4.0]]
X = [[1.0, 10.0], [-1.0, 0.0]]

Y = fp8_dynamic_matmul(W, X)
print(Y.shape)  # (2, 2)
# Y should be close to W @ X, with a small quantization error that grows
# for tensors/tokens whose values span a wide dynamic range.
```

## What the gate checks

Two gates, both against a random weight/activation pair (with a couple of
injected outlier entries, as real tensors have):

- **rel_err** — relative L2 error between your `Y` and the *true*
  full-precision `W @ X`. This just checks your FP8 path is a reasonable
  approximation (threshold is loose: quantization noise is expected).
- **oracle_rel_err** — relative L2 error between your `Y` and a reference
  implementation that runs the *exact same* per-tensor/per-token E4M3
  quantize → matmul → dequantize pipeline described above. This is the
  tight gate: skipping quantization, quantizing `X` per-tensor instead of
  per-token, or using the wrong scale formula will pass the first gate by
  luck (or fail it) but will not match the oracle's specific numbers.
