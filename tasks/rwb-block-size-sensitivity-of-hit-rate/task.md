## Context

A prefix-caching KV-cache server (vLLM's `APC`, SGLang's RadixAttention,
etc.) splits every request's token sequence into fixed-size **blocks**. A
block's KV activations can be reused across requests only if that block's
tokens exactly match a block already computed by an earlier request *and*
every block before it in the sequence matched too — reuse is a property of
the whole prefix, not of an isolated block, since the attention keys/values
for block $i$ depend on everything before it.

Formally, for a request with tokens $t_0 \ldots t_{n-1}$ and block size $B$,
block $i$ covers $t_{iB \ldots \min((i+1)B,\,n)-1}$. Block $i$ is a cache
**hit** iff blocks $0, \ldots, i$ are all hits and the prefix
$t_0 \ldots t_{\min((i+1)B, n)-1}$ was already produced, as an exact block
boundary, by some earlier request. The first block that fails to match
stops all further reuse for the rest of that request.

This makes the cache **block-size sensitive**: if two requests share a
common prefix of length $P$ that is not a multiple of $B$, only
$\lfloor P/B \rfloor \cdot B$ tokens are actually reused — the tail of the
block straddling position $P$ is wasted even though most of it matches,
because the *whole block's* hash/identity depends on tokens past the
divergence point. A smaller $B$ wastes less of that straddling block (at
the cost of more bookkeeping per block in a real system).

## Task

Implement `block_size_reuse_comparison(requests, block_size_a, block_size_b)`:

```python
def block_size_reuse_comparison(requests: list, block_size_a: int, block_size_b: int):
    ...
```

- `requests`: a list of token-id sequences (each a list of ints), to be
  fed **in order** into a prefix cache, as described above.
- `block_size_a`, `block_size_b`: two candidate block sizes.

Simulate two independent caches, one per block size, both processing the
exact same `requests` in the same order. Return
`(reused_tokens_a, reused_tokens_b, better)`:

- `reused_tokens_a` / `reused_tokens_b`: total number of tokens covered by
  cache-hit blocks, summed over all requests, for that block size.
- `better`: `"a"` if `reused_tokens_a > reused_tokens_b`, `"b"` if the
  reverse, `"tie"` if equal.

## Example

```python
req1 = list(range(64))                # 64 arbitrary tokens
req2 = req1[:20] + list(range(100, 130))  # shares exactly the first 20 tokens, then diverges

block_size_reuse_comparison([req1, req2], 4, 16)
# reused_tokens_a: block_size=4 divides 20 evenly -> 5 full blocks (20
#   tokens) of req2 are reused from req1.
# reused_tokens_b: block_size=16 -> only the first block (tokens[0:16])
#   is a clean match; the block covering tokens[16:32] straddles the
#   divergence at token 20 and is a total miss -> only 16 tokens reused.
# -> (20, 16, "a")   # the smaller block size reuses more tokens
```

## What the gate checks

The gate runs several fixed scenarios (a clean mid-block divergence like
the example, fully identical repeated requests, and a batch with no shared
prefixes at all) plus randomly generated request chains from a seeded
generator, where each request shares a random-length prefix with the
previous one before diverging with fresh random tokens.

For every case, the reference simulates both caches exactly as described
above (processing requests strictly in order, a request's own blocks
becoming visible to the cache only after it finishes, reuse stopping at
the first non-matching block) and compares your full `(reused_tokens_a,
reused_tokens_b, better)` tuple against it with exact equality. A solution
that checks whether a block's *tokens* individually appear somewhere in
the cache (instead of requiring the *whole prefix up to that block* to
match) will overcount reuse whenever a later block happens to repeat
earlier content without the full prefix matching, and will disagree with
the oracle as soon as a test case includes a genuine mid-block divergence.
