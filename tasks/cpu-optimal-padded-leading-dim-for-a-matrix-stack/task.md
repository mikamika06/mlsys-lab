## Context

Stack several same-shaped matrices back-to-back in memory, row-major, and
a cache-set-mapping coincidence can silently wreck performance: if a
matrix's byte footprint happens to be an exact multiple of the cache's
*set period* ($\text{line\_bytes} \times \text{num\_sets}$), then the
same row number in every matrix in the stack maps to the exact same
cache set — no matter how many sets the cache actually has. Depth into
the stack becomes depth into *one* set's associativity, and once the
stack is deeper than the set is wide, every access evicts another.

The fix used throughout real BLAS-style libraries is **padding**: widen
each row from `n_cols` to some `ld >= n_cols` elements (the extra columns
are never touched, just wasted space) so the per-matrix stride is no
longer that exact multiple. Even a minimal nudge — enough to shift off
the periodic boundary — spreads consecutive matrices across *different*
sets and eliminates the alias entirely.

## Task

Implement

```cpp
int choose_padded_ld(int n_cols, int M, int elem_bytes, int line_bytes, int num_sets);
```

Return the smallest `ld >= n_cols` such that
`(M * ld * elem_bytes) % (line_bytes * num_sets) != 0`.

## Example

`n_cols=16, M=8, elem_bytes=4, line_bytes=64, num_sets=8`: the set period
is `64*8=512` bytes, and the unpadded stride is `8*16*4=512` — an exact
multiple, `512 % 512 == 0`. `ld=16` is rejected. `ld=17` gives stride
`8*17*4=544`, and `544 % 512 = 32 != 0` — accepted. `choose_padded_ld`
returns `17`: pad every row by just one element.

## What the gate checks

`exact_match` on the returned `ld` and on the miss counts of a fixed
two-pass-per-row traversal (touch row `r` of all 8 matrices, then touch
the same 8 addresses again — modeling a second pass reusing the first
pass's data) over a deterministic 64-byte-line, 8-set, 4-way cache.
Unpadded (`ld=16`), the second pass's reuse is destroyed by the alias and
every one of the 8x8x2 touches misses (`128` total). With the correct
`ld=17`, the first pass's 64 touches are still compulsory misses, but the
second pass's 64 touches all hit — `64` total, a clean 2x reduction, with
zero change to what's actually computed. Off-by-one on the loop bound, or
checking `ld % num_sets` instead of the true stride modulo the byte-period,
returns a different `ld` and changes the printed miss counts.
