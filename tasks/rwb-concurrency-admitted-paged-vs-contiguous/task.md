## Context

A KV cache has a fixed budget of $N$ physical blocks, each holding $B$
token positions. Two ways of admitting requests into that budget give very
different concurrency:

**Contiguous (slot-based) allocation** must reserve enough blocks for the
*worst case* up front, since it never resizes a request's reservation
later: every admitted sequence reserves $\lceil L_{\max}/B \rceil$ blocks,
where $L_{\max}$ is the maximum sequence length the system must support.
The number of sequences that can run concurrently is therefore

$$
\text{max\_concurrent\_contig} = \left\lfloor \frac{N}{\lceil L_{\max}/B \rceil} \right\rfloor,
$$

independent of how long the actual requests turn out to be.

**Paged allocation** (as in vLLM's PagedAttention) reserves blocks lazily,
one per $B$ tokens *actually* used. Given a stream of requests with lengths
$L_0, L_1, L_2, \dots$ (in arrival order), each request $i$ costs
$\lceil L_i / B \rceil$ blocks, and requests are admitted greedily in order
for as long as the running total still fits the budget:

$$
\text{max\_concurrent\_paged} = \max\Big\{k : \sum_{i=0}^{k-1} \left\lceil \frac{L_i}{B} \right\rceil \le N \Big\}.
$$

Because paged allocation only pays for tokens a request actually has (not
the worst-case maximum), it packs far more concurrent sequences into the
same block budget whenever real requests are shorter than $L_{\max}$ — the
central efficiency argument for paged KV caches in production serving
engines.

## Task

Implement `paged_vs_contiguous_concurrency(seqlens, n_blocks, block_size, max_len)`:

```python
def paged_vs_contiguous_concurrency(seqlens, n_blocks, block_size, max_len):
    ...
```

- `seqlens`: a 1-D array/list of positive integer request lengths, in
  arrival order.
- `n_blocks`: total physical block budget $N$.
- `block_size`: tokens per block, $B$.
- `max_len`: the worst-case context length $L_{\max}$ the contiguous
  allocator must reserve for.

Return a tuple `(max_concurrent_paged, max_concurrent_contig)`:

- `max_concurrent_paged`: greedily walk `seqlens` in order, accumulating
  $\lceil L_i/B \rceil$ blocks per request; stop as soon as the next
  request would exceed `n_blocks`, and return how many requests were
  admitted before that point.
- `max_concurrent_contig`: `n_blocks // ceil(max_len / block_size)`
  (integer division) — a fixed number that does not depend on `seqlens` at
  all.

## Example

```python
import numpy as np

seqlens = [10, 20, 15, 90]
result = paged_vs_contiguous_concurrency(seqlens, n_blocks=6, block_size=8, max_len=64)

# paged: ceil(10/8)=2, ceil(20/8)=3 (running total 5),
#        ceil(15/8)=2 would make 7 > 6 -> stop after 2 requests
# contig: ceil(64/8)=8 blocks/request -> 6 // 8 = 0
# result == (2, 0)
```

## What the gate checks

The gate loads a fixed 60-request length distribution from `seqlens.npy`
and evaluates your function against several `(n_blocks, block_size,
max_len)` configurations. For each configuration it recomputes both
quantities independently — the paged value via the same greedy prefix
walk, the contiguous value via the direct floor-division formula — and
compares your returned tuple for exact equality. All configurations must
match exactly (`exact_match == 1.0`). Using the *actual* request lengths
for the contiguous number instead of the fixed worst-case `max_len`, or
admitting requests out of arrival order for the paged number, will not
match the oracle.
