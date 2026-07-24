## Context

A regular store is a **temporal** access: the CPU treats the write as
data you're likely to touch again soon, so it fetches the target line
into the cache (if it isn't already there) and keeps it around under the
normal LRU replacement policy. That's great for data you actually reuse,
but it's a liability for a large one-shot bulk write like `memset`ting a
big buffer you're about to stream out and never read again — every line
that write touches muscles its way into the cache, evicting whatever
*useful*, actually-reused data happened to be resident.

A **non-temporal (streaming) store** — the semantics behind intrinsics
like `_mm_stream_si128` / `movntdq` — sidesteps this entirely: the write
goes through a small write-combining buffer straight to memory and never
allocates a line in the cache at all. It's a pure write with no
read-for-ownership and no LRU insertion, so it can't evict anything.

This task makes that difference visible as a number: warm a small
"useful" working set into the cache, run a big bulk write over a disjoint
region that's several times the cache's capacity, then re-touch the
useful lines and count how many were evicted.

## Task

`sol.hpp` declares a cache hook `void touch(long byte_addr);`, defined in
`main.cpp` on top of a deterministic 4096-byte set-associative LRU cache
(64-byte lines, 16 sets, 4-way). Calling `touch(addr)` is what makes an
access visible to the cache model; not calling it means the access never
happened as far as the model is concerned — exactly like a non-temporal
store never allocating a cache line.

The driver already defines a fixed, correct `streaming_memset` baseline
(it touches nothing, because non-temporal stores don't touch the cache).
Implement the temporal counterpart:

```cpp
void temporal_memset(long base, long nbytes, int line_bytes);
```

which models an ordinary (temporal) bulk write of `nbytes` bytes starting
at byte address `base`: call

```cpp
touch(line_addr(base, line_bytes, k));
```

for every `k` in `[0, nbytes / line_bytes)`, in increasing order, exactly
once each — one `touch()` per line touched, matching what a real
line-by-line temporal store would do to the cache.

## Example

With `line_bytes = 64`, `base = 4096`, `nbytes = 8192`, the region covers
128 lines, `k = 0..127`. `temporal_memset` must call
`touch(line_addr(4096, 64, 0))`, then `touch(line_addr(4096, 64, 1))`,
..., through `touch(line_addr(4096, 64, 127))` — 128 calls total, each
line touched once.

## What the gate checks

The driver warms an 8-line (512-byte) "useful" working set into a fresh
4096-byte cache, then runs a 8192-byte bulk write (2x the cache's
capacity) over a disjoint region — once through the fixed
`streaming_memset` baseline, once through your `temporal_memset` — and
after each pass re-touches the useful lines and counts how many are now
misses (evicted). It prints both counts. The grader compiles `solve.cpp`
with `clang++ -O2 -std=c++20`, runs it, and requires

$$
\mathrm{exact\_match} = 1 \iff \text{both printed eviction counts match the reference}
$$

`evicted_streaming` is fixed by the harness and always comes out `0` —
non-temporal stores never touch the model, so the useful set survives
untouched. The real test is `evicted_temporal`: on this fixture, a
correct line-by-line temporal `memset` evicts all 8 useful lines
(`evicted_temporal=8`), because the 8192-byte write cycles through every
one of the cache's 16 sets 8 times over — far more than the 4 new
insertions per set it takes to push the useful line out under LRU. An
empty (no-op) implementation prints `evicted_temporal=0`, which does not
match and fails the gate.
