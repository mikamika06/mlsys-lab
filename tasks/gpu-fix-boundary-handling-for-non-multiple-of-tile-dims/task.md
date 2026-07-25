## Context

Tiled GEMM assumes `M`, `N`, `K` are clean multiples of the tile size —
but real matrices usually aren't. When they're not, the tiles along the
bottom row, right column, and last K-slice are "ragged": part of the
tile maps to real matrix elements, part of it maps to nothing. The fix
is simple in principle — treat the missing part as zero — but skipping
it entirely (loading unconditionally) doesn't just corrupt the boundary
threads' own output: because every thread in a tile reads shared-memory
values that OTHER threads loaded, one out-of-range load can poison the
dot product for threads whose own position is perfectly in-range.

## Task

`solve.cu` computes a `6x6 = 6x6 * 6x6` GEMM tiled `4x4` — every
dimension is ragged, since `6` isn't a multiple of `4` — but it loads
every tile element unconditionally, with no check against `M`, `N`, or
`K`. Fix the two load lines so each one is guarded:

```cpp
if (grow < M && a_col < K) {
    As[lr*4 + lc] = A[grow*K + a_col];
} else {
    As[lr*4 + lc] = 0.0f;
}
```

and the matching guard (`b_row < K && gcol < N`) for the `Bs` load. Do
**not** change anything else — not the compute loop, not the final
write's own `grow < M && gcol < N` guard, which is already correct.

## Example

For the last (`kt=1`) K-tile with `K=6, TILE=4`: valid `a_col` values are
`4` and `5` (`kt*4 + lc` for `lc=0,1`); `lc=2,3` give `a_col=6,7`, past
`K`. An unguarded load for `lc=2` reads `A[grow*K + 6]` — one full row
past where `A`'s row `grow` actually ends, into row `grow+1`'s data (or
past `A` entirely, into whatever comes next in memory). A guarded load
instead stores `0.0f`, which contributes nothing to the dot product —
exactly as if that column of `A` simply didn't exist.

## What the gate checks

`check.py` parses `solve.cu` with the real CUDA-C frontend, runs it on a
fixed `6x6` random input, and compares the output against `A @ B`
computed directly in numpy. The scratch memory just past `A`, `B`, and
`C`'s real extent is deliberately filled with a large nonzero constant
(`777.0`, not zero) before the kernel runs, so an unguarded out-of-bounds
load reads something visibly wrong instead of coincidentally reading a
zero. It requires

$$
\mathrm{max\_abs\_err} \le 10^{-6}
$$

The shipped `solve.cu` measures `max_abs_err ≈ 1328` — the missing
bounds checks let the poisoned scratch memory leak into nearly every
tile's dot product, since ragged tiles touch it on both the row/col edge
and the K edge.
