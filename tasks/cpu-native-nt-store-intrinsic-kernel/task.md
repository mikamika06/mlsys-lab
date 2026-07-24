## Context

A regular store to memory has to bring the destination's cache line into
the cache first (allocate-on-write), because the CPU assumes you might
read it back soon — that's usually the right bet, but not always. When
you're writing a *large output buffer once* and won't touch it again for
a long time (flushing a computed result, a bulk copy, a streaming write),
allocating cache lines for data you'll never re-read just evicts other,
genuinely useful data. Real CPUs expose a **non-temporal / streaming
store** for exactly this case (x86 `MOVNTPS`/`MOVNTDQ`, or the
write-combining path a streaming ARM store reaches): the write still
lands in memory, correctly and fully, it just never allocates a cache
line.

## Task

`sol.hpp` gives you two store mechanisms into the same modeled 64-byte-line
cache:

- `store_temporal(float* p, float v)` — `*p = v`, AND registers the
  64-byte line containing `p` as touched.
- `store_nontemporal(float* p, float v)` — `*p = v`, WITHOUT touching any
  cache line.
- `dst_lines_touched()` — how many distinct lines have been registered as
  touched since `reset_dst_lines()`.

Implement:

```cpp
void stream_copy(const float* src, float* dst, int n);
```

Copy `src[0..n)` into `dst[0..n)`, value for value. `dst` is written
exactly once and will not be re-read for a long time — write every element
through `store_nontemporal`, never `store_temporal`, so the copy doesn't
evict anything useful from cache.

The driver (`main.cpp`, fixed) builds a 512-element, 64-byte-aligned `src`
array (`src[i] == i * 0.5`), resets the line tracker, calls your
`stream_copy`, and prints the sum of everything actually landed in `dst`
plus how many distinct destination lines got touched.

## Example

```
checksum=65408.000000
dst_lines_touched=0
```

$\sum_{i=0}^{511} 0.5i = 0.5 \cdot \frac{511 \cdot 512}{2} = 65408$ — every
value copied correctly, and since every write went through
`store_nontemporal`, not one of the 512 floats' worth of destination
memory (32 cache lines at 64 bytes each) was ever registered as resident.

Writing the exact same, correct values through `store_temporal` instead
still prints the right checksum but touches all 32 destination lines:

```
checksum=65408.000000
dst_lines_touched=32
```

Same data, same correctness — 32 cache lines of eviction pressure that a
real streaming store would have avoided entirely.

## What the gate checks

The grader compiles `main.cpp` + your file with `clang++ -O2 -std=c++20`,
runs it, and requires both printed numbers to match the same driver linked
against the reference within `max_abs_err <= 1e-6`. Getting the checksum
right by routing every write through `store_temporal` instead of
`store_nontemporal` still fails: `dst_lines_touched` comes out `32`
instead of `0`, and that difference alone blows past the `1e-6` tolerance.
The starter never calls either store function, so `dst` keeps its `-1.0`
sentinel values and prints `checksum=-512.000000`.
