# Context-shift bookkeeping when the window overflows

In the logs of long sessions (an agent loop generating token after token with
no explicit end) one of two things happens: either the runner crashes the
moment the token count in the cache hits `n_ctx`, or the existing "hack" just
chops off a block of the oldest tokens — and the system prompt at the very
start of the sequence goes with it. A few steps later the model quietly
forgets its role and instructions, even though the session doesn't look like
it crashed from the outside.

Separately, there's a capacity-planning report: cache memory was sized as if
every model were plain multi-head attention, when some of the models are
actually GQA with fewer kv-heads. Those models ended up with twice the memory
reserved that they actually need.

Two things need doing: a correct window shift that never touches the
protected prefix, and an honest cache-byte count that shows the difference
between MHA and GQA on the same context.

## What you write

`ctxshift/bookkeeping.py`:

```python
discard_count(n_past, n_keep) -> int
simulate(n_ctx, n_keep, n_tokens) -> dict
```

The cache holds no more than `n_ctx` tokens at a time. The first `n_keep`
tokens that ever landed in the cache (the system prompt, say) are protected —
they can never be evicted, no matter how many shifts happen. In every test
case `n_ctx` is guaranteed to exceed `n_keep` with margin (at least 2), so
there's always something to shift.

`discard_count(n_past, n_keep)` — how many tokens to remove in one shift when
a cache of `n_past` tokens overflows: `n_left = max(0, n_past - n_keep)`,
result is `n_left // 2` (integer division in half: free up half of what's
unprotected, not all of it and not none of it).

`simulate(n_ctx, n_keep, n_tokens)` — run `n_tokens` generation steps, one new
token per step, starting from an empty cache. Tokens get increasing ids: 0,
1, 2, ... in order of appearance. Before each new token: if
`len(resident) + 1 > n_ctx`, compute `discard_count` from the current
`len(resident)` and `n_keep`, evict `resident[n_keep : n_keep + n_discard]`
(the block right after the protected prefix — not the prefix itself and not
the tail), and only then append the new id at the end. Return:

```python
{"resident": [...], "evicted": [...], "shift_events": [...]}
```

where `resident` is the ids still in the cache at the end of the run,
`evicted` is every evicted id across the whole run in eviction order,
`shift_events` is a list of length `n_tokens` where `shift_events[t]` is how
many ids got evicted at step `t` specifically (0 if no shift happened at that
step).

`ctxshift/memory.py`:

```python
kv_cache_bytes(config) -> int
mha_vs_gqa(config) -> dict
```

`config` is `{"n_layers", "n_heads", "n_kv_heads", "head_dim", "n_ctx",
"bytes_per_element"}`. For every context position, a layer holds keys and
values for each kv-head: `2 · n_kv_heads · head_dim · bytes_per_element`
bytes. For the whole model: `kv_cache_bytes = n_layers · n_ctx · 2 ·
n_kv_heads · head_dim · bytes_per_element`.

`mha_vs_gqa(config)` compares this same model as if it were MHA
(`n_kv_heads == n_heads`) against what it actually is (GQA, `n_kv_heads` from
the config) — at the SAME `n_ctx` and the rest of the parameters. Returns
`{"mha_bytes", "gqa_bytes", "saved_bytes"}`, where `saved_bytes = mha_bytes -
gqa_bytes`.

## How it's checked

The grader computes the reference itself: for the shift bookkeeping — across
several `(n_ctx, n_keep, n_tokens)` combinations, for the memory — across
several model configs (including plain MHA, where `n_kv_heads == n_heads`).
The third milestone is yours: you write a test in `tests/test_regression.py`,
and we swap in an implementation of `simulate` that evicts tokens starting
from the very front of the cache, ignoring `n_keep` — i.e. one that leaks the
protected prefix into eviction. Your test needs to catch that.

```
mlsys project start m-implement-context-shift-bookkeeping-when-the-window-
mlsys project grade m-implement-context-shift-bookkeeping-when-the-window- --milestone 1
```
