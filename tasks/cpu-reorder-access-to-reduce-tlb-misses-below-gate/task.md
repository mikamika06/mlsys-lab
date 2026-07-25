## Context

A TLB caches recent page translations, and — like any small cache — it only
helps when the pages a loop keeps revisiting fit inside it. A `256 x 256`
`double` matrix, stored row-major with 8 rows per 16 KiB page, needs 32
pages in total. Summing it **column by column** — for every column, walk
every row — means every single column-pass needs all 32 pages resident at
once, no matter how the matrix is laid out: consecutive elements of a
column are `ld` doubles apart, one whole row stride, so a column visits
every page the matrix has.

If the TLB has fewer entries than that (here, 20), a column-major sweep
can't even finish one column without evicting pages it will need again on
the very next column: by the time column `j` has touched all 32 pages, only
the most recent 20 are still resident, so column `j+1` starts by missing on
the ones that fell out. Every column repeats this from scratch.

The fix doesn't touch the data or the matrix layout at all — only the
**order** accesses happen in. Split the rows into blocks small enough that
one block's pages *do* fit in the TLB, and sweep every column for the
*whole block* before moving to the next block. Now a block's pages get
loaded once (on the block's first column) and stay resident for every later
column in that same block — page revisits cluster together in time instead
of scattering across the whole sweep.

## Task

Implement, in `solve.cpp`:

```cpp
double sum_matrix_reordered(const double* data, int R, int C, int ld);
```

Sum every element of the `R x C` matrix (leading dimension `ld` doubles per
row), calling `touch_page(&data[i*ld + j])` exactly once per element, in
whatever order over the `(i, j)` pairs you choose — the sum must come out
the same regardless of order, only `tlb_miss_count()` afterward is meant to
change. Pick an order that groups nearby rows together across the whole
column sweep (a **row-blocked** traversal: process a block of rows across
*all* columns before moving to the next block of rows) so each block's
pages stay resident in the pinned 20-entry TLB for its entire sweep.

## Example

With `R=256, C=256, ld=256` (32 total 16 KiB pages, 20-entry TLB), the
driver (`main.cpp`, fixed) reports:

```
R=256 C=256 page_bytes=16384 tlb_entries=20
sum=33029125.0
tlb_misses=32
```

`sum=33029125.0` never depends on traversal order. `tlb_misses=32` comes
from blocking the rows into two 128-row blocks (16 pages each, comfortably
under the 20-entry TLB): each block pays exactly 16 cold misses on its
first column and then hits for every one of its other 255 columns, for
`16 * 2 = 32` misses total. The column-major starter prints the same sum
but `tlb_misses=8192` (`32` misses on *every one* of the 256 columns — a
column alone already needs all 32 pages, more than the TLB can hold, so
each column's sweep evicts pages the next column will need all over
again) — 256x more misses for the exact same arithmetic result.

## What the gate checks

The grader compiles `main.cpp` + `solve.cpp` with `clang++ -O2 -std=c++20`,
runs it, and requires the entire printed output — including `tlb_misses=`,
not just `sum=` — to match the reference (`main.cpp` + `ref.cpp`)
byte-for-byte (`exact_match == 1.0`). A block size larger than the TLB can
hold (e.g. blocking by 256 rows — the whole matrix, i.e. no blocking at
all) reproduces the column-major starter's `tlb_misses=8192`; a block size
so small that the traversal effectively becomes row-major changes the sum's
computation order but not its value, yet still has to print exactly `32` to
pass — printing any other miss count fails the gate even with a correct
sum.
