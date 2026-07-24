## Context

An automatic-prefix-cache (vLLM's `APC`, SGLang's RadixAttention, ...)
splits every request's tokens into fixed-size **blocks** and hashes each
block together with everything before it (a *chain hash*, so a block's
identity depends on its entire prefix, not just its own tokens). A block
is a cache **hit** — its KV activations are reused instead of recomputed —
iff that chained identity already exists in the cache from an earlier
request. Because the hash chains, a hit at block $i$ requires every block
before it to have hit too: the moment one block's prefix diverges from
anything cached, that block and every later block in the same request are
**misses**, and get computed (and then inserted into the cache for future
requests to reuse).

For a request with $n$ tokens and block size $B$, it has
$\lceil n / B \rceil$ blocks. Over a whole stream of requests processed in
order, the two numbers that describe cache effectiveness are the total
count of hit blocks (reused, work skipped) and the total count of miss
blocks (computed, real work done).

## Task

Implement `prefix_cache_block_stats(requests, block_size)`:

```python
def prefix_cache_block_stats(requests: list, block_size: int):
    ...
```

- `requests`: a list of tokenized prompts (each a list of ints),
  processed **in order** against one shared cache.
- `block_size`: the fixed block size.

Return `(total_reused_blocks, total_computed_blocks)` summed over every
request in the stream, simulated exactly as described above (chained
prefix identity, reuse stops at the first divergence, a request's own
blocks only become visible to the cache after it finishes).

## Example

```python
req1 = list(range(32))              # 32 tokens, 4 blocks of 8
req2 = req1[:16] + list(range(100, 120))  # shares the first 2 blocks, then diverges

prefix_cache_block_stats([req1, req2], block_size=8)
# req1: nothing cached yet -> all 4 blocks computed
# req2: blocks 0-1 (tokens 0:16) match req1's cached prefix -> 2 hits;
#       block 2 (tokens 16:24) diverges -> miss, and the remaining block
#       is also a miss (reuse stops at the first divergence)
# -> total_reused_blocks = 2, total_computed_blocks = 4 + 2 = 6
```

## What the gate checks

The gate runs several hand-built streams (a clean 2-block-shared-prefix
case like the example, fully identical repeated requests, a batch of
unrelated requests with no shared prefixes, an empty stream, and a
single short request) plus randomly generated request chains from a
seeded generator — some children sharing a random-length prefix with the
previous request before diverging, some starting over completely fresh.

For every case the reference simulates the same chained-prefix cache
(processing requests strictly in order, a request's own blocks becoming
visible only after it finishes, reuse stopping at the first
non-matching block) and returns the two totals. Your `(reused,
computed)` pair is compared to it with exact equality. A solution that
checks each block's hit/miss status independently — e.g. testing whether
*that block's own tokens* (ignoring everything before it) were seen
before, rather than the full chained prefix — will overcount reuse
whenever a later block's tokens happen to repeat elsewhere without the
full prefix actually matching, and disagree with the oracle as soon as a
test case includes a genuine mid-request divergence.
