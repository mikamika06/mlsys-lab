## Context

A write-combining (WC) buffer collects stores to memory before committing
them as a burst. A simplified WC buffer has a fixed number of cache-line
slots. When a slot's line becomes fully written, the buffer issues a **full
flush** containing every byte of that line. When a line is evicted to make
room for another before it was fully written, the buffer issues a **partial
flush** of whatever bytes it did collect.

For a store trace of byte addresses, each address maps to a cache line:

$$\mathrm{line}(a) = \left\lfloor \frac{a}{B} \right\rfloor, \qquad
\mathrm{offset}(a) = a \bmod B$$

where $B$ is `line_bytes`. A line's WC entry is **full** once every offset in
$[0, B-1]$ has been stored to it.

This models the real difference between contiguous streaming stores (which
fill lines completely and flush efficiently) and scattered stores across
more lines than the buffer has slots (which evict half-written lines and
waste bandwidth on partial flushes).

## Task

Implement `wc_flush_stats` (declared in `sol.hpp`):

```cpp
void wc_flush_stats(const long* addrs, int n, int line_bytes, int slots, long* out);
```

Simulate a WC buffer with `slots` active line entries over the `n`-address
store trace `addrs`:

- For each store, if its line has no active entry: if all `slots` entries are
  occupied, evict the OLDEST entry (FIFO, by the order lines were first
  touched) -- count a `full_flush` if that evicted entry was complete
  (every offset recorded), else a `partial_flush`. Then start a fresh entry
  for the new line.
- Record the store's offset into its line's entry. If the entry now has
  every offset in $[0, B-1]$ recorded, flush it immediately (`full_flush`)
  and free the slot.
- After the trace ends, flush every entry still resident, oldest first,
  using the same full/partial rule.

Write `full_flush` into `out[0]` and `partial_flush` into `out[1]`.

## Example

```
addrs = [0, 1, 2, 3, 4, 5, 6, 7], line_bytes = 4, slots = 2
wc_flush_stats(...) -> out[0] = 2, out[1] = 0
```

The first four stores fill line 0 (offsets 0-3), causing one full flush. The
next four stores fill line 1, causing another full flush.

## What the gate checks

`main.cpp` runs 5 fixed store traces (contiguous fills, strided stores that
never complete a line, and scattered addresses spread over more lines than
there are slots) through your `wc_flush_stats` and prints `(full, partial)`
for each. The gate is `exact_match` on that full printed output against the
reference. Forgetting to flush a line the instant it becomes full (instead
of only at eviction/end-of-trace), evicting the newest entry instead of the
oldest, or mislabeling an evicted complete entry as a partial flush will all
change at least one scenario's counts and fail the gate.
