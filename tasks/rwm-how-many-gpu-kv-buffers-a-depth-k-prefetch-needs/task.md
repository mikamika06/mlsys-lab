## Context

Autoregressive GPU inference keeps key-value (KV) cache blocks for tokens that
will be consumed by the decoder. A prefetcher can hide transfer latency by
loading future KV blocks before they are needed.

For a lookahead depth $K$, the runtime needs one buffer for the block currently
being consumed and $K$ additional buffers for the prefetched future blocks.
Therefore the required number of rotating KV buffers is derived as

$$B = 1 + K.$$

A correct schedule keeps the current buffer available while future buffers are
filled. At each decoding step, the consumer uses one buffer and the prefetcher
advances the next $K$ positions. If fewer than $K+1$ buffers exist, the consumer
eventually reaches a buffer that has not completed its prefetch.

## Task

Implement `kv_buffer_plan(prefetch_depth)`:

```python
def kv_buffer_plan(prefetch_depth: int) -> tuple[int, bool]:
    ...
```

The function receives a non-negative integer lookahead depth $K$ and returns:

1. The minimum number of KV buffers required.
2. Whether a repeating schedule using that many buffers is stall-free.

The returned buffer count must follow the production scheduling rule for
depth-$K$ prefetch: keep the current buffer plus all $K$ prefetched buffers.
The stall-free flag should describe whether that buffer count can support the
lookahead schedule.

## Example

```python
buffers, ok = kv_buffer_plan(3)

# buffers == 4
# ok == True
```

## What the gate checks

The gate generates several prefetch depths and builds an oracle schedule that
simulates the current consumer position and the $K$ future prefetched positions.
It computes the minimum buffer count from the schedule requirement and checks
that the returned count and stall-free flag exactly match the oracle result.
