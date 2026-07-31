# Head Index Map for GQA

The model was moved to our own inference backend without flash-attention
kernels — plain numpy on CPU, for debugging quality issues. On the old
MQA model (one kv-head shared by all) the output matches the original
token-for-token. On the new GQA model (32 query heads, 8 kv-heads) the text
got noticeably worse, even though all the tensor shapes check out: the cache
is the right size, and there are exactly as many heads as in the config.
Someone, while porting, expanded the kv-heads to match the number of
query-heads the wrong way — and no shape check catches it, because the
shape comes out right either way.

We need to explicitly build a map of "which query-head reads which kv-head",
compute oracle attention on top of it, and write a test that catches a swap
of the correct grouping for a wrong one.

## What you write

`gqa/mapping.py`:

```python
build_head_map(num_q_heads, num_kv_heads) -> list[int]
build_query_groups(num_q_heads, num_kv_heads) -> list[list[int]]
```

`num_q_heads` always divides evenly by `num_kv_heads`; `group = num_q_heads
// num_kv_heads` is how many query-heads map to one kv-head. The convention
is contiguous blocks: query-heads `[0, group)` read kv-head `0`,
`[group, 2*group)` read kv-head `1`, and so on (this is how weights are
actually laid out in real GQA models: the kv-head is expanded into `group`
consecutive copies, not "every n-th query-head"). `build_head_map` returns a
list of length `num_q_heads` where element `q` is the index of the kv-head
that query-head `q` reads. `build_query_groups` returns a list of length
`num_kv_heads` where element `k` is the sorted list of query-heads belonging
to kv-head `k`.

`gqa/attention.py`:

```python
expand_kv(kv, num_q_heads) -> np.ndarray
gqa_attention(q, k, v, num_kv_heads) -> np.ndarray
```

`kv` is an array of shape `(num_kv_heads, seq, head_dim)` (keys or values
separately). `expand_kv` expands it to `(num_q_heads, seq, head_dim)` using
the same block convention as `build_head_map`: row `q` of the result is row
`build_head_map(num_q_heads, num_kv_heads)[q]` of the input array.
`gqa_attention` computes plain causal scaled-dot-product attention: `q` has
shape `(num_q_heads, seq, head_dim)`, `k`/`v` have shape `(num_kv_heads,
seq, head_dim)`; each query-head only sees positions `j <= i` of the
kv-head assigned to it by the map.

## How it's graded

The grader computes the oracle itself, with an independent implementation
(loops instead of vectorization), across several head configurations and
several random (fixed-seed) attention cases. The third milestone is yours:
you write a test in `tests/test_regression.py`, and we swap `build_head_map`
for a variant that hands out query-heads round-robin (`q % num_kv_heads`)
instead of contiguous blocks. Your test must catch that.

```
mlsys project start m-gqa-head-index-map
mlsys project grade m-gqa-head-index-map --milestone 1
```
