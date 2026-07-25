## Context

Modern CPUs execute SIMD vector loads and scalar (gather) loads with very
different micro-op (uop) counts. A **contiguous, aligned** vector load
that reads `vec_width` elements issues exactly **one** load uop, no
matter how many bytes that register spans -- the address is computed
once. A **gather** load that fetches the same number of elements from
scattered indices has no vector hardware support for the address
pattern, so the CPU computes each element's address separately and
issues **one load uop per element**.

That uop-count gap (`n` vs `n / vec_width`) is only half the story.
Whether those loads are cheap or expensive in cycles also depends on
**cache traffic**, which this task models with a real, deterministic
64-byte-line / 32-set / 4-way LRU cache (fed through a `touch()` hook,
since real hardware cache timing isn't reproducible). A sequential scan
touches every 64-byte line exactly once -- the theoretical minimum. A
gather over the *same set of elements*, visited in a badly-ordered
permutation, can touch a line, get evicted before the line's other
elements are read, and pay for that same line again later -- turning a
128-line working set into far more than 128 misses, purely from
visitation order.

## Task

Implement, in `solve.cpp`:

```cpp
long contiguous_load(long base, int n, int vec_width, int elem_bytes);
long gather_load(long base, const int* idx, int n, int elem_bytes);
```

`contiguous_load` reads `n` elements (an exact multiple of `vec_width`)
strictly in order, `vec_width` elements per chunk. Call `touch()` exactly
once per chunk, at the chunk's first byte address
(`base + chunk_index * vec_width * elem_bytes`), and return the number
of chunks (`n / vec_width`) as the modeled uop count.

`gather_load` reads the same `n` elements through a permutation `idx`:
`idx[k]` is the k-th element index visited. Call `touch()` exactly once
per element, in order, at `base + (long)idx[k] * elem_bytes`, and return
`n` as the modeled uop count -- one load uop per element.

`touch()`, `reset_cache()`, and `misses()` are declared in `sol.hpp` and
defined in `main.cpp`: a fixed 64-byte-line, 32-set, 4-way LRU cache
model. You only need to call `touch()` at the right addresses, in the
right order, the right number of times -- the cache model does the rest.

## Example

The driver fixes `n = 4096`, `vec_width = 16`, `elem_bytes = 4` (so one
vector chunk is exactly `16 * 4 = 64` bytes -- one cache line), and
builds `idx` as a 12-bit bit-reversal permutation of `0..4095` (the same
trick FFT implementations use to produce a maximally scattered visitation
order over a fixed index set).

Running the reference:

```
contig_uops=256 contig_misses=256
gather_uops=4096 gather_misses=4096
```

Both passes touch the identical 4096 elements / 256 distinct 64-byte
lines. The contiguous pass needs only 256 load uops and 256 misses (the
compulsory minimum -- every line is fetched once and never revisited).
The gather pass needs 4096 load uops (16x more, exactly `vec_width`) —
and its scattered visitation order means every single touch misses too:
by the time a line's other 15 elements are visited, it has already been
evicted. Same elements, same cache, only the order differs, and it costs
4096 misses instead of 256. (Visiting `idx[k] = k`, i.e. the same order
as `contiguous_load`, through `gather_load` reproduces the 256-miss
result -- the miss count is a property of visitation order, not of
"gather" as such.)

## What the gate checks

The grader compiles `main.cpp` + `solve.cpp` with `clang++ -O2 -std=c++20`,
runs it, and requires `exact_match == 1` against the same driver linked
with `ref.cpp`: both printed lines (`contig_uops=... contig_misses=...`
and `gather_uops=... gather_misses=...`) must match exactly. The starter
returns `0` from both functions and never calls `touch()`, so it prints
`contig_uops=0 contig_misses=0` / `gather_uops=0 gather_misses=0` --
wrong uop counts and, since the cache model was never fed a single
address, wrong (zero) miss counts too.
