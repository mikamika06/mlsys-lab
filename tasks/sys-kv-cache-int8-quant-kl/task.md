## Context

Serving long-context LLMs, the key/value (KV) cache is often the dominant
memory cost. A common trick is to store the cache in **int8** instead of
fp16/fp32: each cached token's key vector (and value vector) is quantized
independently with a symmetric per-token scale,

$$
s_i = \frac{\max_j |K_{ij}|}{127}, \qquad
\hat{K}_{ij} = \mathrm{round}\!\left(\frac{K_{ij}}{s_i}\right) \cdot s_i ,
$$

clipping the rounded code to $[-127, 127]$ before it is stored as an
`int8`. The same scheme is applied to $V$.

This shrinks the cache by \~4x versus fp32, but the dequantized keys
$\hat K$ are no longer exactly $K$, so the attention logits computed from
$\hat K$,

$$
\text{logits} = \frac{Q\,\hat K^{\top}}{\sqrt{d}},
$$

and therefore the attention-weight distribution
$\mathrm{softmax}(\text{logits})$, drift slightly from what an
unquantized fp32 cache would produce. The question this task measures:
*how much* does that distribution drift, and how much does the final
attention output drift?

## Task

Implement `kv_cache_int8_attention`:

```python
def kv_cache_int8_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray):
    ...
```

* `Q` — `(m, d)` fp32 array of query vectors.
* `K`, `V` — `(n, d)` fp32 arrays: the keys/values that would be written
  into the KV cache.

Your implementation must:

1. Quantize `K` and `V` to `int8` using the **per-token (per-row)
   symmetric** scheme above (a separate scale per row, computed from that
   row's max absolute value).
2. Dequantize both back to floating point.
3. Compute standard single-head scaled dot-product attention against the
   dequantized cache:
   $\text{logits} = Q\hat K^\top / \sqrt d$,
   $w = \mathrm{softmax}(\text{logits})$,
   $\text{out} = w\hat V$.

Return `(logits, out)`: `logits` has shape `(m, n)` (the pre-softmax
scores), `out` has shape `(m, d)` (the attention output).

## Example

```python
import numpy as np
rng = np.random.default_rng(0)
Q = rng.standard_normal((2, 8)).astype(np.float32)
K = rng.standard_normal((5, 8)).astype(np.float32)
V = rng.standard_normal((5, 8)).astype(np.float32)

logits, out = kv_cache_int8_attention(Q, K, V)
print(logits.shape, out.shape)   # (2, 5) (2, 8)
```

## What the gate checks

The grader computes a plain fp32 reference (no quantization at all) for
the same `Q, K, V`: `ref_logits = Q @ K.T / sqrt(d)` and the standard
softmax-weighted `ref_out`.

* **`mean_kl`** — mean KL divergence between `softmax(ref_logits)` and
  `softmax(logits)` (your returned logits), i.e. how much the
  attention-weight distribution drifted because `K` was quantized. Must
  satisfy $\mathrm{mean\_kl} \le 10^{-2}$.
* **`rel_err`** — relative L2 error between `out` and `ref_out`, i.e. how
  much the final attention output drifted because *both* `K` and `V` were
  quantized. Must satisfy $\mathrm{rel\_err} \le 5\times10^{-2}$.

A correct per-token int8 quantizer keeps both metrics small (quantization
error is a small fraction of each token's own magnitude). A dequantization
bug — e.g. forgetting to multiply the int8 codes back by their scale —
leaves the codes on a completely different numeric scale than the
original vectors and fails both gates by orders of magnitude.
