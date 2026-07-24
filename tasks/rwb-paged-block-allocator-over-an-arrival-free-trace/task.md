## Context

A KV cache serving pool has $N$ physical blocks, each holding $B$ token
positions. Requests arrive, hold their allocation while they run, and
depart (freeing it) — a real allocator has to handle this whole lifecycle,
not just a one-shot admission decision.

**Paged allocation** (as in vLLM's PagedAttention) treats every block as
interchangeable: a request needing $L$ tokens costs
$\lceil L/B \rceil$ blocks from a shared free pool, taken at arrival and
returned at departure. Because blocks aren't tied to a specific physical
layout, any free block can satisfy any request — a simple free-block
counter is all the bookkeeping needed.

**Contiguous (worst-case) allocation** must instead reserve
$\lceil L_{\max}/B \rceil$ blocks per request — the same fixed amount
regardless of that request's actual length — since it can't grow a
reservation later.

Given the same arrival/departure trace, run both policies and compare:

- $\text{peak\_blocks\_used}$ — the most blocks the **paged** allocator
  ever had in use at once (the real memory high-water mark).
- $\text{admitted\_count}$ — how many requests the **paged** allocator
  accepted (versus rejecting because the pool was full at their arrival).
- the same admitted count, but for the **contiguous** allocator's
  worst-case-sized reservations.

## Task

Implement `paged_allocator_trace(arrive_t, depart_t, seq_len, n_blocks, block_size, max_len)`:

```python
def paged_allocator_trace(arrive_t, depart_t, seq_len, n_blocks: int, block_size: int, max_len: int) -> tuple[int, int, int]:
    ...
```

- `arrive_t`, `depart_t`, `seq_len`: equal-length sequences, one entry per
  request — arrival timestamp, departure timestamp
  (`depart_t[i] > arrive_t[i]`), and tokens of KV cache needed while alive.
- `n_blocks`: total physical block pool size $N$.
- `block_size`: tokens per block $B$.
- `max_len`: the worst-case context length $L_{\max}$ a contiguous
  allocator must reserve for, per request.

Build the combined, time-ordered sequence of arrival and departure
events. **On a timestamp tie, process departures before arrivals** — a
block freed at time $t$ is available to a request arriving at that same
$t$.

Run **two** independent simulations over this same event order with a
simple free-block counter (starting at `n_blocks`):

1. **Paged**: at each arrival, the request needs
   $\lceil \text{seq\_len}/B\rceil$ blocks; admit it (deduct from the
   free count) only if that many are free right now, else reject it (its
   later departure event then does nothing). At an admitted request's
   departure, return its blocks. Track `peak_blocks_used` — the largest
   the "blocks in use" count ever reaches.
2. **Contiguous**: identical mechanics, but every request's cost is the
   fixed $\lceil L_{\max}/B\rceil$, ignoring its actual `seq_len`.

Return `(peak_blocks_used, admitted_count_paged, admitted_count_contiguous)`.

## Example

```python
arrive_t = [0, 0, 5]
depart_t = [10, 4, 12]
seq_len  = [20, 8, 12]
paged_allocator_trace(arrive_t, depart_t, seq_len, n_blocks=4, block_size=8, max_len=32)
# t=0: request 0 needs ceil(20/8)=3 blocks -> admitted (free 4->1, used=3, peak=3)
#      request 1 needs ceil(8/8)=1 block -> admitted (free 1->0, used=4, peak=4)
# t=4: request 1 departs -> returns 1 block (free 0->1, used=3)
# t=5: request 2 needs ceil(12/8)=2 blocks -> only 1 free -> REJECTED
# t=10: request 0 departs -> returns 3 blocks
# t=12: request 2's departure event does nothing (it was never admitted)
# paged: peak_blocks_used=4, admitted_count_paged=2
#
# contiguous: every request costs ceil(32/8)=4 blocks -- the whole pool.
# t=0: request 0 admitted (free 4->0). request 1 needs 4 -> REJECTED (free=0).
# t=5: request 2 needs 4 -> REJECTED (free still 0).
# t=10: request 0 departs, returns all 4.
# admitted_count_contiguous=1
```

## What the gate checks

The gate loads a fixed 25-request arrival/departure trace from
`arrive_t.npy`/`depart_t.npy`/`seq_len.npy` (heavy overlap, so the pool
genuinely fills up and some requests get rejected while others are
admitted later once earlier ones depart), evaluated against several
`(n_blocks, block_size)` configurations, plus several fully independent
seeded synthetic traces. For each configuration it reruns both
simulations independently and compares your returned triple for exact
equality (`exact_match`, fraction of cases matching, gate requires
`1.0`). Processing arrivals before departures on a tie, forgetting to
free a rejected request's (nonexistent) allocation, or reusing the
paged cost function for the contiguous comparison will diverge from the
oracle on at least one configuration.
