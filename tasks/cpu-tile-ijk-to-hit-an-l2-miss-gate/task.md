## Context

A naive triple-nested `ijk` matmul (`i` outer, `j` middle, `k` inner) has
good locality for `A` (row `i` is swept contiguously as `k` advances) and
for `C` (`C[i][j]` is the same address for the entire inner `k` loop, so
it stays hot), but terrible locality for `B`: as `k` advances, `B[k][j]`
strides by a full row (`N` elements) every single step, touching a
different cache line almost every time. Once the matrices are bigger than
the cache, that column-striding sweep through `B` dominates the miss
count — and it repeats, from scratch, for every one of the `N` values of
`i`.

**Tiling** (blocking) all three loops into `T x T x T` chunks fixes this:
finish every `(i, j, k)` triple inside one tile before moving to the
next, so each tile's `T x T` sub-blocks of `A`, `B`, and `C` — small
enough to fit in cache — are fully reused before anything evicts them,
instead of being swept once (mostly missing) per outer iteration.

## Task

Implement

```cpp
void tiled_matmul(int N, int T);
```

Touch `a_addr(N,i,k)`, `b_addr(N,k,j)`, `c_addr(N,i,j)` for every `(i,j,k)`
in `[0,N) x [0,N) x [0,N)`, **exactly once each**, but visit the index
space tile-by-tile (steps of `T` in each of `i`, `j`, `k`) rather than one
flat sweep.

## Example

For `N=48`, `T=8`: the naive sweep re-strides all the way through `B`'s
column `j` for every value of `i` — 48 separate cold sweeps through a
matrix bigger than the whole cache. The tiled version only ever has an
`8x8` slice of `A`, `B`, and `C` live at once (`3 * 8 * 8 * 4 = 768`
bytes — comfortably cache-resident), reusing each loaded element across
the other two tiled dimensions before moving on.

## What the gate checks

`exact_match` on `(naive_misses, tiled_misses)` for a fixed 48x48 matmul
over a 64-byte-line, 32-set, 4-way (8192-byte) cache. Reference:
`naive_misses=4735`, `tiled_misses=1147` — better than a 4x reduction from
tiling alone, with the exact same computation. Touching addresses in the
wrong order within a tile, tiling only one or two of the three loops, or
touching an `(i,j,k)` triple more than once, changes the printed miss
count.
