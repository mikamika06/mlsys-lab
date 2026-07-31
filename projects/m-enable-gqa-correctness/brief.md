# GQA grouping in scaled_dot_product_attention

Profiled inference and noticed: the new model has several times more query
heads than kv heads (grouped-query attention). Instead of physically
duplicating the K/V cache for every query head, we turned on the `enable_gqa`
flag in attention — each query head just shares the cache with its group, and
KV-cache memory really did drop by several times.

But regression on long dialogues (somewhere past a hundred tokens of context)
got worse: the model sometimes answers as if it mixed up who belongs to which
group. On short prompts, and on configs where every query head has its own kv
head (no grouping), there's no difference at all. Looks like it's specifically
the decomposition of query heads into groups that breaks, whenever grouping
actually groups something.

## What you write

`gqa/expand.py` — `repeat_kv(x, n_rep) -> np.ndarray`. `x` has shape
`(batch, kv_heads, seq, head_dim)`. Each kv head needs to be expanded into
`n_rep` consecutive query heads: output head `i` must take its data from
exactly kv head `i // n_rep`. Result shape: `(batch, kv_heads * n_rep, seq, head_dim)`.
At `n_rep == 1` this is the identity.

`gqa/mask.py` — `causal_bias(q_len, kv_len, dtype) -> np.ndarray`, shape
`(q_len, kv_len)`. The causal mask is defined only for `q_len == kv_len`
(context prefill, not incremental decoding). Position `(i, j)` equals `0.0`
if `j <= i` (that key can be attended to), otherwise `-inf`.

`gqa/core.py` — `scaled_dot_product_attention(query, key, value, is_causal=False, scale=None, enable_gqa=False) -> np.ndarray`.
`query` has shape `(batch, q_heads, L, head_dim)`, `key`/`value` —
`(batch, kv_heads, S, head_dim)`. If `enable_gqa=True`, `q_heads` must divide
evenly by `kv_heads`; before the computation, `key` and `value` are expanded
via `repeat_kv` to `q_heads` heads. After that it's ordinary attention:
`softmax(query @ key.transpose(-1, -2) * scale + bias) @ value`, where `scale`
defaults to `1 / sqrt(head_dim)`, and `bias` comes from `causal_bias` if
`is_causal`, otherwise zero. Key requirement: heads assigned to different
kv heads must not affect one another — changing the contents of one kv head
must change the output of exactly the query heads assigned to it, and no
others.

## How it's graded

The grader computes the reference itself — an independent numpy
implementation of the same formula across several configs (different
batch/heads/lengths, with and without causality). The third milestone is
yours: write a test in `tests/test_regression.py`, and we'll swap in a
head-expansion where query heads map to kv heads not in consecutive blocks
but interleaved (every n-th head belongs to the same kv head). Your test
needs to catch that.

```
mlsys project start m-enable-gqa-correctness
mlsys project grade m-enable-gqa-correctness --milestone 1
```
