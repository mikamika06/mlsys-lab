## Context

Scaled dot-product attention with an additive bias (a padding mask, an
ALiBi slope $\times$ distance term, a relative-position bias, ...) computes

$$
\text{logits} = \left(\frac{QK^\top}{\sqrt{d}}\right) + B, \qquad
\text{attn} = \operatorname{softmax}(\text{logits})\, V,
$$

where $B \in \mathbb{R}^{n_q \times n_k}$ is added **after** the
$1/\sqrt{d}$ scaling, and is itself left unscaled. This is exactly the
formula real libraries use (e.g. PyTorch's
`scaled_dot_product_attention` with a float `attn_mask`): the mask/bias
represents a fixed logit offset — for instance "subtract a huge number to
mask this key" or "shift this logit by an ALiBi slope" — and must not be
shrunk by the attention scale factor.

A common bug folds the bias into the pre-scale product instead:

$$
\text{logits}_{\text{buggy}} = \left(QK^\top + B\right) \cdot \frac{1}{\sqrt{d}}
= \frac{QK^\top}{\sqrt{d}} + \frac{B}{\sqrt{d}} .
$$

Every bias term silently shrinks by the scale factor. For a masking bias
of $-10^9$ this rarely matters (still effectively $-\infty$), but for
smaller, meaningful biases — ALiBi slopes, learned relative-position
terms, soft masks — the attention weights come out measurably wrong.

## Task

Fix `sdpa_with_additive_bias`:

```python
def sdpa_with_additive_bias(q: np.ndarray, k: np.ndarray, v: np.ndarray, bias: np.ndarray, scale: float) -> np.ndarray:
    ...
```

* `q` — shape $(n_q, d)$
* `k` — shape $(n_k, d)$
* `v` — shape $(n_k, d_v)$
* `bias` — shape $(n_q, n_k)$, additive logit bias
* `scale` — the scalar applied to $QK^\top$ (typically $1/\sqrt{d}$, but
  passed in explicitly — do not assume it)

Compute $QK^\top$, multiply by `scale`, **then** add `bias` unscaled,
then apply a numerically stable softmax over the last axis and multiply
by `v`. Return shape $(n_q, d_v)$.

## Example

```python
import numpy as np

q = np.array([[1.0, 0.0]])
k = np.array([[1.0, 0.0], [0.0, 1.0]])
v = np.array([[10.0], [0.0]])
bias = np.array([[0.0, 5.0]])   # strongly favor the second key
scale = 1.0 / np.sqrt(2)

out = sdpa_with_additive_bias(q, k, v, bias, scale)
# The bias of 5.0 is large relative to the ~0.7-scale logits, so most of
# the softmax mass goes to the second key/value even though q dotted with
# the first key is larger before biasing.
```

## What the gate checks

A single gate, **max_abs_err**, compares your output against an fp64
NumPy oracle that scales $QK^\top$ first and adds the bias afterward,
across several random-shape cases plus an ALiBi-style linear-bias case
and a zero-bias sanity case. The buggy ordering (`(QK^\top + B) *
scale`) produces errors well above $10^{-2}$ on the biased cases; the
correct ordering must bring `max_abs_err` down to `<= 1e-5`.
