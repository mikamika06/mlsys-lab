## Context

A compiler like XLA or TVM often faces a three-stage elementwise
pipeline

$$
D = (A + B) \odot C, \qquad E = \max(D, 0) - A, \qquad F = \sum_i E_i,
$$

written naively as three separate full-array passes that each
materialize an intermediate array ($D$, then $E$, then reduce to $F$).
**Loop fusion** rewrites this as one pass: process the arrays in
**tiles**, and for each tile compute $D$, then $E$, then a partial sum
of $F$ — all before moving to the next tile — so no full-size
intermediate array is ever fully materialized at once. Because $+,\odot,
\max(\cdot,0), -$ are all elementwise and the sum is associative, tiling
the loop this way must produce (up to floating-point summation order)
the exact same $E$ and $F$ as the naive three-pass version — that
numerical equivalence is exactly what licenses the compiler to do the
fusion.

## Task

Implement `fused_tile_pipeline(A, B, C, tile_size)`:

```python
def fused_tile_pipeline(A: list[float], B: list[float], C: list[float], tile_size: int):
    ...
```

- `A, B, C`: 1-D float arrays, all the same length `n`.
- `tile_size`: positive int, the number of elements processed per tile.
  `n` need not be a multiple of `tile_size` — the last tile is whatever
  remains.

For each contiguous tile `A[start:end]` (`end = min(start+tile_size,
n)`), compute that tile's `D`, `E = max(D,0) - A`, and its partial sum
of `E`, fused in one pass over the tile, before advancing to the next
tile. Return `(E, F)`: `E` is the full `(n,)` array (every tile's
`E`-slice written into the matching positions), `F` is the scalar
`sum(E)` as a Python float (accumulated tile-by-tile, not from a
separate final pass over `E`).

## Example

```python
A = [1.0, -2.0, 3.0, 0.5]
B = [0.0,  1.0, -1.0, 2.0]
C = [2.0,  1.0,  1.0, 1.0]
E, F = fused_tile_pipeline(A, B, C, tile_size=3)
# D = [2.0, -1.0, 2.0, 2.5]
# E = relu(D) - A = [1.0, 0.0, -1.0, 2.0]
# F = 2.0
```

## What the gate checks

The grader runs 12 deterministic cases (`random` seeded)
with varying array length and `tile_size` — including lengths that are
an exact multiple of `tile_size` and lengths that leave a remainder
tile — against a baseline computed with plain full-array Python ops
(`D=(A+B)*C; E=[max(d, 0) for d in D]-A; F=sum(E)`). `max_abs_err <= 1e-6`,
taken as the worse of the elementwise error on `E` and the scalar error
on `F`. Dropping or double-processing elements at a tile boundary,
mishandling the final short tile, or forgetting to subtract `A` inside
the fused tile (rather than after) will show up as a mismatch.
