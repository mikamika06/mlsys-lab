## Context

Transposing a matrix, `out[j][i] = in[i][j]`, has a structural cache
problem: whichever loop order you pick, ONE of the two arrays gets walked
with a large stride. Row-major `i` outer, `j` inner reads `in` contiguously
but writes `out` with stride `N` — every write jumps to a different row.
If the matrix doesn't fit in cache, that stride means the line you just
wrote to `out` is long gone by the time you come back to its row, and it
has to be fetched all over again.

**Cache blocking** doesn't remove the stride — it can't, transposition is
inherently a stride-N operation on one side — but it confines the damage to
a small `B x B` tile at a time: process every `(i, j)` pair inside one tile
completely before moving to the next tile. While a tile is being processed,
the handful of `out` rows it touches stay small enough to stay
cache-resident, so the strided writes land in lines that are still there
from a moment ago, instead of having been evicted and needing a full
refetch.

## Task

Implement

```cpp
void blocked_transpose(int N, int B);
```

which computes `out[j][i] = in[i][j]` for an `N x N` matrix by touching
`in_addr(N,i,j)` then `out_addr(N,j,i)` (both declared in `sol.hpp`) for
every `(i, j)` pair, **exactly once each** — but iterating the index space
in `B x B` **tiles**: for each tile `(ii, jj)`, finish every `(i, j)` pair
with `ii <= i < ii+B` and `jj <= j < jj+B` before moving to the next tile.

## Example

For `N = 64`, `B = 8`: the first tile covers `i in [0,8)`, `j in [0,8)` —
all 64 `(i,j)` pairs inside it are processed (`in`/`out` touched for each)
before the loop moves to the next `j`-tile, `j in [8,16)`, still with
`i in [0,8)`.

## What the gate checks

The driver transposes a 64x64 matrix twice — once with a fixed (harness)
naive row-by-row loop, once with your `blocked_transpose` — each against
its own fresh 8192-byte cache model (`in` and `out` together are 32768
bytes, 4x the cache's capacity), and prints both miss counts. The grader
compiles `solve.cpp` with `clang++ -O2 -std=c++20`, runs it, and requires

$$
\mathrm{exact\_match} = 1 \iff \text{both printed miss counts match the reference}
$$

`naive_misses` is fixed by the harness and always matches — the real test
is `blocked_misses`. On this fixture the naive loop measures 4352 misses;
a correctly `8x8`-tiled traversal measures 768 — better than 5x fewer, from
changing nothing but the order the exact same `(i,j)` pairs are visited in.
