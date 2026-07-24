## Context

Historically, PyTorch's CUDA caching allocator grows by calling `cudaMalloc`
for a brand-new, independent **segment** each time no existing free block is
big enough for a request. Each segment occupies its own separate virtual
address range. Free blocks can only coalesce with a *neighbor in the same
segment* — two freed blocks that happen to be the same size, in different
segments, can never merge into one bigger block, no matter how much total
free memory exists. This is **external fragmentation**: plenty of free
bytes, but scattered across segments in pieces too small individually for a
new large request, forcing yet another segment (and eventually an
out-of-memory error) even when the *sum* of free memory would clearly be
enough.

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` fixes this by reserving
one contiguous virtual address range up front and mapping physical pages
into it on demand as the allocator grows — so the entire arena behaves like
a *single* segment. Anything freed anywhere in that range can coalesce with
its neighbor regardless of when each part was mapped in, eliminating this
kind of external fragmentation.

## Task

Implement `replay_trace(trace, capacity, expandable=False)`:

```python
def replay_trace(trace, capacity, expandable=False) -> dict:
    ...
```

- `trace`: a list of operations, each `("alloc", name, size)` or
  `("free", name)`, `name` a unique string, `size` a positive int.
- `capacity`: total device bytes available (an int).
- `expandable`: `False` simulates the legacy fixed-segment allocator;
  `True` simulates `expandable_segments:True`.

**Fixed segments (`expandable=False`).** Free blocks live inside whichever
segment created them. On `alloc(name, size)`, search the free blocks of
*all* segments for one with `size` $\geq$ requested; among such candidates
pick the smallest (best-fit; ties broken by lower segment id, then lower
offset within that segment) and split off the remainder as a new free block
in that *same* segment. If no existing free block fits, create a brand-new
segment sized exactly to the request: if `reserved + size > capacity` the
allocation fails (OOM) and the trace stops immediately; otherwise the new
segment becomes fully occupied by this allocation and `reserved` grows by
`size`. On `free(name)`, return the block to its own segment's free list,
merging with an adjacent free block *in that same segment* (by offset) as
long as merges apply — freed blocks in different segments never merge.

**Expandable segments (`expandable=True`).** There is conceptually one
segment: a bump-allocated arena. `alloc`/`free` behave exactly as above
(best-fit, split, grow-on-miss, coalesce-on-free), except every block lives
in this single arena, so coalescing on `free` can merge across regions that
were mapped in at different times.

Return `{"oom": bool, "peak_reserved": int}`: whether any `alloc` failed,
and the highest `reserved` value reached (from just before the failure, if
any — `reserved` never changes on a failed grow).

## Example

```python
trace = [
    ("alloc", "A", 700),
    ("alloc", "B", 700),
    ("free", "A"),
    ("free", "B"),
    ("alloc", "C", 1400),
]

replay_trace(trace, capacity=2000, expandable=False)
# A and B each land in their own 700-byte segment. Freeing both leaves two
# same-size free blocks in two DIFFERENT segments -- they cannot merge. C
# needs 1400 contiguous bytes, which no single segment offers, so a third
# segment must be created: reserved would reach 700+700+1400=2800 > 2000.
# {"oom": True, "peak_reserved": 1400}

replay_trace(trace, capacity=2000, expandable=True)
# A and B are two halves of the same growing arena. Freeing both leaves two
# ADJACENT free blocks that coalesce into one 1400-byte block, which
# exactly satisfies C with no further growth.
# {"oom": False, "peak_reserved": 1400}
```

## What the gate checks

The gate implements the same two-mode allocator independently as an oracle
and replays several fixed traces, each once with `expandable=False` and once
with `expandable=True`, comparing the submission's `(oom, peak_reserved)`
pair in both modes against the oracle's. The traces are built so the two
modes genuinely disagree on the OOM outcome, exercising the exact
cross-segment coalescing gap that `expandable_segments` closes.
