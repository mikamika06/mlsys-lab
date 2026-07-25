## Context

A tokenizer sharded across `NUM_THREADS = 8` worker threads, each keeping
its own local histogram of `NUM_BINS = 4` token-category counts (merged
into a single global histogram only once, at the very end) sounds
perfectly parallel — no thread ever reads or writes another thread's
counters. But a cache doesn't track individual variables, it tracks
64-byte *lines*. If two threads' counter blocks are packed close enough
together that they land on the same line, every write from either thread
forces a real MESI-coherent cache to invalidate the OTHER thread's cached
copy of that line and re-fetch exclusive ownership — **false sharing**:
memory that is logically private causes cross-core cache traffic anyway,
purely from physical proximity.

Each thread's block is `NUM_BINS * 8 = 32` bytes of real data (4 `int64`
counters). Pack 8 threads' blocks back-to-back with no padding and pairs
of threads (`0`&`1`, `2`&`3`, ...) land on the same 64-byte line — every
time execution ping-pongs between the two threads sharing a line, that
line gets invalidated and re-acquired.

## Task

`sol.hpp` gives you a deterministic coherence model:

- `write_counter(thread_id, addr)` — a write to `addr` on behalf of
  `thread_id`. If the 64-byte line containing `addr` was last written by
  a DIFFERENT thread, that's an invalidation.
- `invalidation_count()` — total invalidations since `reset_coherence()`.

Implement:

```cpp
size_t thread_block_stride();
```

Return the byte stride between the start of thread `t`'s block of
`NUM_BINS` counters and thread `(t+1)`'s block (the driver places thread
0's block at a 64-byte-aligned base address, thread `t`'s block at
`base + t * thread_block_stride()`), chosen so that no two threads' blocks
ever land on the same 64-byte line.

The driver (`main.cpp`, fixed) runs a deterministic 20-round, round-robin
schedule: every round, threads `0..7` each bump all 4 of their own local
bins by 1, in order. It prints your stride, the total bytes used
(`NUM_THREADS * stride`), the invalidation count, and a checksum of every
counter's final value (which never depends on the layout — only the
coherence traffic does).

## Example

Padding each thread's 32 live bytes up to a full 64-byte line
(`stride = 64`, `512` bytes total: half of it padding, none of it shared):

```
stride=64
total_bytes=512
invalidations=0
checksum=640
```

Zero invalidations: every thread owns entire, exclusive cache lines for
the whole run. Packing tightly instead (`stride = 32`, the natural
"no wasted bytes" choice) puts 2 threads per line for 4 of the 8 lines:

```
stride=32
total_bytes=256
invalidations=156
checksum=640
```

Same checksum ($8 \times 4 \times 20 = 640$ increments total, always) —
156 invalidations purely from two threads sharing physical cache lines
they never logically share any data on.

## What the gate checks

The grader compiles `main.cpp` + your file with `clang++ -O2 -std=c++20`,
runs it, and requires every printed number to `exact_match` the same
driver linked against the reference stride. Halving memory by packing
tightly gets the checksum right but reports `invalidations=156` instead of
`0` and `total_bytes=256` instead of `512` — both wrong, and the gate
fails. Only a stride that is an exact multiple of 64 bytes eliminates the
cross-thread line sharing entirely.
