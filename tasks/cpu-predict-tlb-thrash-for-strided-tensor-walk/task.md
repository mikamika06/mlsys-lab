## Context

A TLB (Translation Lookaside Buffer) caches recent virtual-to-physical page
translations. Like any small cache, it only helps if the pages a loop keeps
revisiting fit inside it. **TLB reach** — `tlb_entries * page_bytes` — is
the largest working set of memory a loop can walk before earlier pages start
getting evicted to make room for new ones.

The trap is stride. Walking a `4096`-column, row-major, `float32` matrix
*by column* — element `i` at address `i * row_bytes` — moves `row_bytes`
per step, not `4` bytes. If a page is `16384` bytes and a row is exactly
`16384` bytes wide, **every single element lands on a brand-new page**: a
1000-row column touches 1000 distinct pages from a matrix that, walked
row-major, would only ever need one page resident at a time. The data
hasn't changed size — only the *order* of access has, and that alone can
blow straight through TLB reach.

The opposite trap is estimating the working set from the address **span**
instead of counting pages exactly. A stride that is a multiple of the page
size *skips* pages entirely — a dilated view that jumps 2 pages at a time
touches exactly one page per element, never the page in between, so its
true working set is half of what `span / page_bytes` would suggest.

## Task

Implement, in `solve.cpp`:

```cpp
TlbVerdict classify_tlb_thrash(int extent, long long stride_bytes, int elem_bytes,
                                int page_bytes, int tlb_entries);
```

The walk visits `extent` elements at addresses `i * stride_bytes` for `i` in
`[0, extent)`. Compute the number of **distinct pages** touched — one page
per element, using `floor(i * stride_bytes / page_bytes)`, counting each
distinct value exactly once (an exact set, not an approximated
`span / page_bytes`). Return `TlbVerdict::Thrash` if that count exceeds
`tlb_entries` (a repeated sweep cannot keep every page resident), otherwise
`TlbVerdict::NoThrash`.

## Example

The driver (`main.cpp`, fixed) pins a 16 KiB page size and a 128-entry TLB,
and runs five walks:

```
contiguous_small extent=100000 stride=4 verdict=0
contiguous_huge extent=10000000 stride=4 verdict=1
column_walk_page_stride extent=1000 stride=16384 verdict=1
large_stride_skips_pages extent=100 stride=32768 verdict=0
small_stride_fits extent=500 stride=64 verdict=0
```

`contiguous_small` (100000 float32s, stride 4 — a plain sequential scan)
spans about 390 KB, only ~25 pages: `NoThrash`. `contiguous_huge` is the
same pattern at 10,000,000 elements (~2400 pages): `Thrash`.
`column_walk_page_stride` walks a column of a 4096-wide `float32` matrix —
stride exactly one page — so 1000 elements means 1000 distinct pages:
`Thrash`, despite touching far fewer total *bytes* of real data than
`contiguous_small`. `large_stride_skips_pages` (100 elements, stride =
2 pages) touches exactly 100 distinct pages — well under 128 — even though
its address span covers 200 pages' worth of memory: `NoThrash`.
`small_stride_fits` (500 elements, stride 64) spans only about 2 pages:
`NoThrash`.

## What the gate checks

The grader compiles `main.cpp` + `solve.cpp` with `clang++ -O2 -std=c++20`,
runs it, and requires the entire printed output to match the reference
(`main.cpp` + `ref.cpp`) byte-for-byte (`exact_match == 1.0`). Approximating
distinct pages as `ceil(extent * stride_bytes / page_bytes)` (a span-based
estimate that assumes every page inside the span is touched) gets the three
contiguous/near-contiguous cases right but is wrong on
`large_stride_skips_pages`: the span-based estimate is `200` pages
(`Thrash`, since `200 > 128`), while the exact distinct-page count is only
`100` (`NoThrash`) — the skipped pages in between were never counted out.
