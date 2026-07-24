## Context

PyTorch's CUDA caching allocator carves memory out of a single growing arena.
When a `malloc(n)` request arrives it looks for a free block big enough to
satisfy it (splitting off the leftover as a new free block), and only grows
the arena's *reserved* high-water mark on a genuine miss. Adjacent free
blocks are coalesced back together when freed.

Left unconstrained, a large freed block can get carved into many small
pieces to satisfy unrelated small requests. If a *large* allocation shows up
again later, no single leftover fragment may be big enough, forcing the
arena to grow — and if the device is already near capacity, that growth
fails with an out-of-memory (OOM) error, even though the *sum* of free bytes
would have been plenty.

`PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:X` addresses exactly this: free
blocks larger than $X$ bytes may **not** be split to serve a smaller
request. They can still be reused, but only by a request whose size matches
the block exactly. This sacrifices reuse for small requests (which now have
to grow the arena themselves) in exchange for keeping large blocks intact
for future large requests of the same size.

## Task

Implement `replay_trace(trace, capacity, max_split_size=None)`:

```python
def replay_trace(trace, capacity, max_split_size=None) -> dict:
    ...
```

- `trace`: a list of operations, each `("alloc", name, size)` or
  `("free", name)`, where `name` is a unique string identifying a live
  allocation and `size` is a positive int.
- `capacity`: the arena's maximum size in bytes (an int).
- `max_split_size`: `None`, or a positive int cap.

Simulate a single-arena allocator over `trace`:

1. **alloc(name, size).** Search the free blocks for one whose size is
   `>= size`. A candidate is usable when `max_split_size is None`, or its
   size is `<= max_split_size`, or its size is exactly `== size` (an
   oversized block may still be reused by an exact-size request even under
   the cap). Among usable candidates pick the smallest (best-fit; break
   ties by the lowest offset). If a fitting block is found and it is larger
   than `size`, carve off the first `size` bytes and keep the remainder as
   a new free block; otherwise no split is needed.
   If no usable free block exists, grow the arena: if
   `reserved + size > capacity`, the allocation **fails** — record the
   failure, stop processing the trace immediately (no later operations
   run), and report the arena as having hit OOM. Otherwise place the new
   block at the current `reserved` offset and increase `reserved` by
   `size`.
2. **free(name).** Return that block to the free list. Merge it with an
   immediately adjacent free block on either side (by offset), repeating
   until no more merges apply.

Return `{"oom": bool, "peak_reserved": int}`: whether any `alloc` in the
trace failed, and the highest value `reserved` ever reached (the value from
just before an OOM, if one occurred — a failed grow never changes
`reserved`).

## Example

```python
trace = [
    ("alloc", "A", 1000),
    ("free", "A"),
    ("alloc", "s1", 100),
    ("alloc", "s2", 100),
    ("alloc", "s3", 100),
    ("alloc", "s4", 100),
    ("alloc", "B", 1000),
]

replay_trace(trace, capacity=1500, max_split_size=None)
# -> without a cap, freeing A's 1000-byte block lets s1..s4 carve it up;
#    when B needs 1000 bytes again no single fragment is big enough, so the
#    arena must grow by another 1000 -> 2000 > 1500 -> OOM.
# {"oom": True, "peak_reserved": 1000}

replay_trace(trace, capacity=1500, max_split_size=500)
# -> with the cap, A's 1000-byte block (> 500) cannot be split for the
#    100-byte requests, so s1..s4 each grow the arena instead (+400 total);
#    A's block survives untouched and B (size 1000, an exact match) reuses
#    it directly with no further growth.
# {"oom": False, "peak_reserved": 1400}
```

## What the gate checks

The gate implements the same allocator independently as an oracle and
replays several fixed traces, each once with `max_split_size=None` and once
with a specific cap, comparing the submission's `(oom, peak_reserved)` pair
in both runs against the oracle's. The traces are built so that the two
runs genuinely disagree on the outcome — exercising the exact fragmentation
trade-off the cap exists to fix.
