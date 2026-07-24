## Context

A write-combining (WC) buffer collects stores to memory before committing them as a burst. A simplified WC buffer has a fixed number of cache-line slots. When a slot becomes full, the buffer can issue a full flush containing all bytes in that line. When another line must replace a partially filled slot, the remaining bytes are written by a partial flush.

For a store trace of byte addresses, each address maps to a cache line:

$$\mathrm{line}(a) = \left\lfloor \frac{a}{B} \right\rfloor$$

where $B$ is the cache-line size in bytes. A line is full when every byte offset in the range $[0, B-1]$ has been stored.

This task models the difference between contiguous streaming stores and scattered stores. The model also sends the store trace through a deterministic cache simulator. The cache has fixed parameters, so the result depends only on the generated access sequence.

## Task

Implement `wc_flush_stats(addrs, line_bytes, slots)`:

```python
def wc_flush_stats(addrs: list[int], line_bytes: int, slots: int) -> tuple[int, int]:
    ...
```

The function receives a byte-address store trace. Simulate a WC buffer with `slots` active cache-line entries.

For each store:
- Add the byte offset `addr % line_bytes` to the WC entry for that line.
- If the line is already full after the store, count one `full_flush` and remove the entry.
- If a new line needs a WC entry while all slots are occupied, evict the oldest entry. Count a `full_flush` if the evicted entry is complete, otherwise count a `partial_flush`.
- At the end of the trace, flush all remaining entries. Complete entries count as `full_flush`; incomplete entries count as `partial_flush`.

Return `(full_flush, partial_flush)`.

## Example

```python
trace = [0, 1, 2, 3, 4, 5, 6, 7]
wc_flush_stats(trace, 4, 2)
# (2, 0)
```

The first four stores fill line $0$, causing one full flush. The next four stores fill line $1$, causing another full flush.

## What the gate checks

The gate runs several deterministic store traces through both the candidate implementation and an internal WC reference model. It also runs the traces through a fixed cache simulator with `line_bytes=64`, `sets=8`, and `ways=2` to ensure the generated access behavior is evaluated deterministically.

The returned tuple must exactly match the reference result:

$$
(\mathrm{full\_flush}, \mathrm{partial\_flush})
$$

Any mismatch fails the gate.
