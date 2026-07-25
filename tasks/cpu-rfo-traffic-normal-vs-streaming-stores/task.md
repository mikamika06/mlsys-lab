## Context

Caches operate at the granularity of a whole line, not individual bytes.
When an ordinary ("temporal") store writes to a line that isn't already
cache-resident, the cache controller can't just write the few bytes the
instruction touches -- MESI-style coherence requires it to hold the
*entire* line in the Modified/Exclusive state before any byte of it can
change. So it first issues a **Read-For-Ownership (RFO)**: a full-line
read from memory, purely to get exclusive ownership, even though every
byte of that line is about to be overwritten by the store anyway. Later,
when that now-dirty line is evicted, the whole line is written back to
memory too. Net traffic per line: one full-line **read** (the wasted
RFO) plus one full-line **write** (the eventual writeback).

**Non-temporal (streaming) stores** (`_mm_stream_ps` / `movnt*` on
x86, similar streaming-store intrinsics elsewhere) skip the cache
entirely. Since the data never becomes cache-resident, there is nothing
to gain ownership of and nothing to write back later -- the store goes
straight to memory through write-combining buffers. Net traffic per
line: one full-line **write**, period.

For a pure write-only sweep over a buffer with no reuse (the case where
non-temporal stores are the right tool), this makes ordinary stores cost
**twice** the memory-bus traffic of streaming ones, for identical
program behavior.

## Task

Implement:

```cpp
long temporal_store_traffic(long total_bytes);
long nontemporal_store_traffic(long total_bytes);
```

Using the pinned `LINE_BYTES` (declared in `sol.hpp`, defined in
`main.cpp`): round `total_bytes` up to a whole number of `LINE_BYTES`
lines, `lines = ceil(total_bytes / LINE_BYTES)`.

- `temporal_store_traffic`: `lines * LINE_BYTES * 2` (RFO read +
  writeback, per line).
- `nontemporal_store_traffic`: `lines * LINE_BYTES` (write only, per
  line).

## Example

With `LINE_BYTES = 64` and `total_bytes = 1024`: `lines = 1024 / 64 =
16`. `nontemporal_store_traffic = 16 * 64 = 1024` -- exactly the buffer
size, since every byte is written exactly once and nothing else moves.
`temporal_store_traffic = 1024 * 2 = 2048` -- double, from the RFO read
of every line before it could be overwritten.

## What the gate checks

`main.cpp` runs five fixed buffer sizes (from `1` byte to `100000`
bytes, including sizes that aren't exact multiples of `LINE_BYTES`)
through both functions and prints `temporal`, `nontemporal`, and their
ratio for each. The candidate's full stdout is compared byte-for-byte
(`exact_match = 1.0`) against the reference's, whose ratio is always
exactly `2.0`. Treating a temporal store as costing the same traffic as
a non-temporal one (forgetting the RFO read) makes every printed ratio
`1.0` instead and fails.
