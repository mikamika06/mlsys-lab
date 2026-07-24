## Context

FP8 E4M3 has a fixed maximum representable magnitude, $448$. Quantizing a
per-head key cache $K \in \mathbb{R}^{n \times d}$ needs one scale $s$ so
that $K/s$ fits the format; the standard choice is

$$
s_{\text{amax}} = \frac{\max_{i,j} |K_{ij}|}{448} .
$$

This is exact for the single largest element, but a **single outlier
key** can inflate $s_{\text{amax}}$ far past what the rest of the cache
needs — every other key then gets quantized with a coarser step than its
own magnitude would require. An alternative uses a high percentile of
the magnitudes instead of the true max:

$$
s_{p} = \frac{\operatorname{percentile}(|K|,\ p)}{448}, \qquad p = 99.5,\ 99.9,\ \dots
$$

Any element whose magnitude exceeds $s_p \cdot 448$ **saturates** — it
gets clipped to $\pm 448 \cdot s_p$ instead of being represented
exactly. That is a real cost, but it is only paid by the outlier(s); if
an outlier key is one that the query barely attends to anyway (its
attention logit $q \cdot K_i / \sqrt d$ is far below the max logit, so
$\operatorname{softmax}$ assigns it negligible weight), clipping it away
costs almost nothing at the *output* — while every other key benefits
from a scale that actually matches its own dynamic range.

## Task

Implement `per_head_scale_attention_errors`:

```python
def per_head_scale_attention_errors(K: np.ndarray, V: np.ndarray, q: np.ndarray, percentile: float) -> np.ndarray:
    ...
```

* `K`, `V` — shape $(n, d)$, fp64 key/value cache for one head.
* `q` — shape $(d,)$, fp64 query vector.
* `percentile` — percentile in $[0, 100]$ used for the percentile scale.

1. Compute $s_{\text{amax}} = \max(|K|) / 448$ and
   $s_p = \operatorname{percentile}(|K|, \texttt{percentile}) / 448$.
2. Quantize $K$ to FP8 E4M3 (round-to-nearest-even against the real
   E4M3 grid, saturating at $\pm 448$) and dequantize, once with each
   scale, giving $\hat K_{\text{amax}}$ and $\hat K_p$.
3. Compute exact softmax attention,
   $\operatorname{softmax}(Kq/\sqrt d)^\top V$, once with the true `K`
   (baseline) and once each with $\hat K_{\text{amax}}$ and $\hat K_p$
   in place of `K` (`V` stays exact fp64 throughout, isolating the
   effect of the K-scale choice).

Return `np.array([attn_max_abs_err_amax, attn_max_abs_err_percentile])`,
the max absolute error of each variant's attention output against the
exact baseline.

## Example

```python
import numpy as np

rng = np.random.default_rng(0)
K = rng.standard_normal((200, 8))
V = rng.standard_normal((200, 8))
q = rng.standard_normal(8)
q_hat = q / np.linalg.norm(q)

# a single key pointed strongly AWAY from q: huge magnitude (dominates
# amax) but a deeply negative logit (gets ~0 softmax weight regardless)
K[0] = -6000.0 * q_hat

out = per_head_scale_attention_errors(K, V, q, percentile=99.5)
# out[1] (percentile scale) < out[0] (amax scale)
```

## What the gate checks

The gate, **max_abs_err**, compares your 2-element array against an
fp64 NumPy oracle across several hand-picked `(K, V, q, percentile)`
cases, each built with one deliberately huge, deeply-off-query outlier
key. The oracle confirms (via assertion) that the percentile-clipped
scale always yields a strictly lower attention output error than the
amax scale on these cases. Your result must match the oracle to within
`1e-9`; using the same scale for both entries, or swapping which one is
amax vs percentile, fails the gate.
