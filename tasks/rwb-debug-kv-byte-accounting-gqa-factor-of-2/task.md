## Context

The KV cache stores **keys and values** for every layer, every position,
and every **KV head** — not every attention (query) head. Under
grouped-query attention (GQA) or multi-query attention (MQA),
`num_kv_heads` is smaller than `num_attention_heads` (a group of query
heads shares one KV head), which is the entire point: it's what shrinks
the cache. A correct byte-accounting formula must:

1. use `num_kv_heads`, not `num_attention_heads`, and
2. count **both** K and V (a factor of 2) — omitting either mistake
   silently overstates savings or undercounts real memory pressure.

$$
\text{bytes} = 2 \cdot \text{batch\_size} \cdot \text{seq\_len} \cdot \text{num\_kv\_heads} \cdot \text{head\_dim} \cdot \text{num\_layers} \cdot \text{bytes\_per\_elem}
$$

## Task

`starter.py` contains a KV-cache byte-accounting function with two bugs:
it multiplies by `num_attention_heads` instead of `num_kv_heads` (so it's
blind to GQA/MQA — reporting the same bytes regardless of how few KV
heads the model actually has), and it drops the factor of 2 for storing
both K and V (only counts one of them).

Fix `kv_cache_bytes(config, seq_len, batch_size)`:

```python
def kv_cache_bytes(config: dict, seq_len: int, batch_size: int) -> int:
    ...
```

- `config`: a dict with keys `num_attention_heads`, `num_kv_heads`,
  `head_dim`, `num_layers`, `bytes_per_elem` (all positive ints).
- `seq_len`, `batch_size`: positive ints.

Return the total KV-cache size in bytes, per the formula above.

## Example

```python
config = {
    "num_attention_heads": 32, "num_kv_heads": 8,
    "head_dim": 128, "num_layers": 32, "bytes_per_elem": 2,
}
kv_cache_bytes(config, seq_len=4096, batch_size=1)
# 2 * 1 * 4096 * 8 * 128 * 32 * 2 = 549,755,813,888 / ... (exact int)
# Using num_attention_heads=32 instead of num_kv_heads=8 would overstate
# this by 4x; dropping the factor of 2 would understate it by half.
```

## What the gate checks

The grader builds several `config` dicts from a seeded NumPy generator
(MHA where `num_kv_heads == num_attention_heads`, MQA with
`num_kv_heads = 1`, and GQA with a few different group sizes, plus
varying `seq_len`, `batch_size`, `num_layers`, and `bytes_per_elem` for
both fp16 and int8) and computes the reference byte count independently
from the formula above using exact integer arithmetic, never calling
your function or hardcoding an expected number.

`exact_match` is the fraction of configs where your returned byte count
equals the oracle's exactly, and the gate requires `1.0`. Using
`num_attention_heads` anywhere in the formula produces the right answer
only for the MHA configs (where the two head counts coincide) and a
wrong one for every GQA/MQA config; dropping the factor of 2 is wrong on
every config.
