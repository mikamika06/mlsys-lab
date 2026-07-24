## Context

Matrix multiply, `C = A*B`, touches every element of `A` and `B` far more
than once — the classic flat `i-j-k` loop revisits a whole row of `B` for
every single element of `C` it produces. If the matrices don't fit in
cache, that reuse is wasted: by the time the loop comes back around to a
value, it has long since been evicted.

**Loop tiling (blocking)** fixes this by restricting the loop to work on a
small `tile x tile x tile` sub-cube of the `(i, j, k)` iteration space at
a time, so the handful of `A`/`B`/`C` values that sub-cube touches stay
cache-resident for its whole duration instead of getting evicted and
re-fetched. **Two-level tiling** nests a second, smaller tile inside the
first — one tile sized for a big cache level, a smaller tile inside it
sized for a smaller/faster level — squeezing out reuse a single tile size
can't capture on its own.

## Task

Implement

```cpp
void matmul_miss_triple(int N, int tile1, int tile2,
                         long a_base, long b_base, long c_base, long* out);
```

which runs THREE loop orders over the *same* `N x N` matmul, `A`/`B`/`C`
addressed via `addr()` (declared in `sol.hpp`), each against its own
freshly `reset_cache()`-ed cache, writing `miss_count()` right after each
into `out[0..2]`:

- `out[0]`: flat `i-j-k` triple loop, no tiling.
- `out[1]`: single-level tiling, `tile1 x tile1 x tile1` tiles.
- `out[2]`: two-level tiling, `tile1 x tile1 x tile1` outer tiles, each
  further split into `tile2 x tile2 x tile2` inner tiles.

All three touch `addr(A,i,k)`, then `addr(B,k,j)`, then `addr(C,i,j)`, for
every `(i, j, k)` — same set of addresses, same total count, only the
*order* differs between variants.

## Example

For `N=8, tile1=4`: the 1-level-tiled variant finishes every `(i, j, k)`
with `i, j, k` all in `[0, 4)` before moving to the next tile, `k in
[4, 8)` (still `i, j in [0, 4)`) — 8 tiles total, each internally a flat
triple loop over its `4 x 4 x 4` sub-cube.

## What the gate checks

`main.cpp` runs two fixed scenarios — a `64x64` matmul with `tile1=16,
tile2=8`, and a `32x32` matmul with `tile1=8, tile2=4` — against a fixed
4096-byte (64-byte line, 16-set, 4-way) cache, and prints the miss triple
for each. The candidate's full stdout is compared byte-for-byte
(`exact_match = 1.0`) against the reference's. On the `64x64` fixture the
reference measures `naive=266880`, `tiled1=71664`, `tiled2=9312` — nearly
a 29x reduction from naive to two-level tiling, from touching the exact
same addresses in a different order. Reusing the same tile size for
`tile1` and `tile2`, or nesting the loops in the wrong order (e.g.
sweeping the outer tiles' `k` before their `j`), still visits every
address the right *number* of times but in the wrong order, changing
which lines are still resident when they're needed again.
