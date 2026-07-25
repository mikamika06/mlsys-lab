## Context

A TLB caches virtual-to-physical page translations, not data — but it
has the same finite-capacity, LRU-eviction behavior as any other cache,
just at 4096-byte page granularity instead of 64-byte line granularity.
The "TLB reach" (entries x page size) is the biggest working set that
can stay translated without re-walking the page table. Sweeping a matrix
in an order that keeps the live set of touched pages inside that reach
pays for each page translation once; an order that keeps jumping far
enough to evict pages before coming back re-walks them over and over.

## Task

Implement

```cpp
double sum_matrix_tlb_friendly(const double* values, long base, int R, int C);
```

Visit every element of the `R x C` row-major matrix exactly once, in
**row-major** order (`r` outer, `c` inner), `touch_page()`-ing each
element's simulated address (`base + (r*C + c) * 8`) and accumulating
its real value (`values[r*C + c]`) into the sum you return.

## Example

For a 64x256 matrix of doubles (131072 simulated bytes = 32 pages)
against an 8-entry, 4096-byte-page TLB (reach = 32768 bytes = 8 pages):
row-major order touches each of the 32 pages exactly once: **32**
misses. Column-major order (`c` outer, `r` inner) sweeps all 64 rows for
a single column before moving to the next — which alone touches all 32
pages of the matrix, blowing well past the 8-page reach — so the *next*
column starts from a TLB that has evicted everything and re-walks all 32
pages again: **8192** misses (32 pages x 256 columns), 256x worse, for
the identical sum.

## What the gate checks

`exact_match`: the driver prints the sum and TLB miss count for a fixed
64x256 matrix. Column-major (or any other) traversal order gives the
same sum but a dramatically higher miss count, so the printed line fails
to match; a starter returning `0.0` fails outright.
