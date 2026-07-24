## Context

A KV cache stores every past key and value vector, and at long context
lengths it dominates inference memory. A common trick is to quantize the
cache to int8 with one scale **per token** (per row of $K$ and $V$), since
each token's key/value vector has its own typical magnitude.

For a matrix $x \in \mathbb{R}^{n \times d}$ (rows = tokens), per-token
symmetric absmax int8 quantization is

$$
s_i = \frac{\max_j |x_{i,j}|}{127}, \qquad
q_{i,j} = \operatorname{clip}\!\left(\operatorname{round}\!\left(\frac{x_{i,j}}{s_i}\right), -127, 127\right),
$$

and the dequantized row is $\hat{x}_{i,j} = q_{i,j} \cdot s_i$.

Quantizing $K$ and $V$ this way and immediately dequantizing them, then
running standard scaled dot-product attention

$$
\operatorname{Attn}(Q, \hat{K}, \hat{V}) = \operatorname{softmax}\!\left(\frac{Q\hat{K}^{\top}}{\sqrt{d}}\right)\hat{V}
$$

lets you measure exactly how much the int8 KV cache degrades the attention
output compared to running the same query against the full-precision $K,
V$.

## Task

Implement `int8_kv_attention(Q, K, V)`.

- `Q`: shape `(n_q, d)`
- `K`: shape `(n_kv, d)`
- `V`: shape `(n_kv, d_v)`

Steps:

1. Quantize `K` and `V` row-wise with the per-token symmetric absmax int8
   scheme above (if a row is all zero, use `scale = 1.0` instead of
   dividing by zero), producing dequantized `K_hat`, `V_hat`.
2. Compute the int8-path attention output:
   `out = softmax(Q @ K_hat.T / sqrt(d)) @ V_hat`.
3. Compute the full-precision attention output from the un-quantized `K`,
   `V`: `full = softmax(Q @ K.T / sqrt(d)) @ V`.
4. Compute `mse = mean((out - full) ** 2)`.

Return `(out, mse)`, where `out` is a `float64` array of shape
`(n_q, d_v)` and `mse` is a Python `float`.

## Example

```python
import numpy as np

rng = np.random.default_rng(0)
Q = rng.normal(size=(3, 8))
K = rng.normal(size=(6, 8))
V = rng.normal(size=(6, 4))

out, mse = int8_kv_attention(Q, K, V)
# out.shape == (3, 4); mse is small but strictly positive
```

## What the gate checks

The gate rebuilds the identical per-token int8 quantize/dequant + attention
pipeline with an independent NumPy oracle across several `(Q, K, V)` shapes,
including all-zero rows in `K` or `V`.

- `rel_err`: the relative L2 error between your `out` and the oracle's
  attention output must be at most `1e-5`.
- `mse`: your reported `mse` must match the oracle's to within `1e-10`.

A solution that quantizes `K`/`V` with a single tensor-wide scale instead of
one scale per row, or that computes `mse` against the quantized-path output
instead of the full-precision output, will diverge from the oracle on both
gates.
