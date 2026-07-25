## Context

A cache line holds several consecutive elements (16 doubles-as-float32
here, at 4 bytes each in the simulated address space, in a 64-byte
line). Visiting a row-major matrix in row-major order — advancing the
column index fastest — touches every element of a line before moving on,
so each line is fetched once and reused fully. Visiting the SAME matrix
with the column index outer and the row index inner jumps a full row's
width (256 bytes = 4 lines) between consecutive touches: elements that
share a line are now visited many iterations apart, by which point a
small cache has already evicted that line for something else — every
touch becomes a fresh miss.

## Task

Implement

```cpp
double sum_matrix(const double* values, long base, int R, int C);
```

Visit every element of the `R x C` matrix exactly once, in **row-major**
order (`r` outer, `c` inner) — matching how it is actually laid out —
`touch_byte()`-ing each element's simulated address
(`base + (r*C + c) * 4`) and accumulating its real value (from
`values[r*C + c]`) into the sum you return.

## Example

For a 64x64 matrix (16384 simulated bytes) against the 512-byte cache
model here, row-major order touches each of the 256 distinct lines
exactly once: **256** misses. Column-major order (`c` outer, `r` inner)
revisits the same 256 lines, but each one is now touched once per column
sharing it (16 columns per line) with 63 *other* rows' lines evicting it
in between every single time: **4096** misses — 16x more, for the exact
same sum.

## What the gate checks

`exact_match`: the driver prints the sum and the miss count for a fixed
64x64 matrix. Column-major (or any other) traversal order gives the
identical sum but a much higher miss count, so the printed line fails to
match even though the "answer" looks right; a starter returning `0.0`
fails outright.
