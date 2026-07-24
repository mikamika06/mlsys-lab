## Context

A naive transpose (`out[j][i] = in[i][j]` for every `i, j` in row-major
order) strides across an entire row (or column) of the matrix for every
single element written, which is terrible for cache locality on a large
matrix. A **blocked** (tiled) transpose processes the output in small
`block x block` tiles instead: for each tile, both the slice of `in` and
the slice of `out` it touches stay within a small working set.

Blocking only changes the ORDER elements are visited in — never their
values. A correct blocked transpose must be byte-for-byte identical to the
naive one.

## Task

Implement `blocked_transpose(const double* in, double* out, int n, int block)`
(declared in `sol.hpp`): write `out[j * n + i] = in[i * n + j]` for every
`i, j` in `[0, n)`, but do it tile by tile — loop over tile row `bi` and
tile column `bj` in steps of `block`, then over `i` in `[bi, bi+block)` and
`j` in `[bj, bj+block)` inside each tile. `n` is always an exact multiple of
`block`.

## Example

For the fixed 12x12 input (`in[i][j] = (i*12 + j) * 0.5`, `block = 4`), the
correct output's first two rows are:

```
0.000 6.000 12.000 18.000 24.000 30.000 36.000 42.000 48.000 54.000 60.000 66.000
0.500 6.500 12.500 18.500 24.500 30.500 36.500 42.500 48.500 54.500 60.500 66.500
```

— column `j` of the output holds what was row `j` of the input, exactly as
a naive transpose would produce; only the ORDER those 144 writes happen in
differs. A starter that never writes `out` leaves it at its seeded `-1.0`
sentinel everywhere.

## What the gate checks

The grader compiles `main.cpp` + `solve.cpp` with `clang++ -O2 -std=c++20`,
runs it, and requires an **exact match** of the full printed 12x12 output
against the same driver linked with `ref.cpp`. Getting a tile boundary
wrong (e.g. transposing `in[i][j]` into `out[i][j]` within a tile instead of
`out[j][i]`) reorders or duplicates values and fails the gate.
