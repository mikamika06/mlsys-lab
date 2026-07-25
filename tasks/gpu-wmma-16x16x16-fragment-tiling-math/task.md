## Context

A tensor-core `16x16x16` matrix-multiply-accumulate (WMMA) doesn't
produce its $16 \times 16 = 256$-element output tile in one place — it
splits it across the 32 lanes of a warp, 8 elements per lane, in a fixed
hardware-defined pattern (a "fragment"). Before that tile can be used as
an ordinary row-major matrix — written to global memory, fed into the
next op — every lane's 8 fragment elements have to be scattered out to
their real `(row, col)` position.

This task defines its own simplified fragment-to-element mapping (not
meant to reproduce any specific real GPU's exact PTX layout, just to
practice the derivation): lane `t` owns two 4-wide runs of columns, one
in row `t/4`, and the same 4 columns again 8 rows further down, in row
`t/4 + 8`:

$$\text{row}(t, k) = \left\lfloor \frac{t}{4} \right\rfloor + 8 \left\lfloor \frac{k}{4} \right\rfloor \qquad \text{col}(t, k) = (t \bmod 4) \cdot 4 + (k \bmod 4)$$

for `k` in `[0, 8)`. Every one of the 256 `(row, col)` cells is covered
by exactly one `(t, k)` pair — a genuine bijection, no cell missed, none
written twice.

## Task

Write a CUDA-C kernel, launched as one warp (32 threads):

```cpp
__global__ void wmma_store_c(float* out, const float* frag);
```

`frag[t*8 + k]` (for `k` in `[0, 8)`) is lane `t`'s `k`-th accumulator
element. For every `k` from `0` to `7`, compute `row`/`col` from the
formulas above and write `out[row*16 + col] = frag[t*8 + k]`.

## Example

Lane `t = 5`: `t/4 = 1`, `t%4 = 1`. For `k = 0..3`
(`k/4 = 0`, `k%4 = 0..3`): `row = 1`, `col = 4, 5, 6, 7`. For `k = 4..7`
(`k/4 = 1`, `k%4 = 0..3`): `row = 9`, `col = 4, 5, 6, 7`. Lane 5 owns
`out[1][4..7]` and `out[9][4..7]` — two 4-element column runs, 8 rows
apart, matching no other lane's 8 cells.

## What the gate checks

The grader parses your `.cu` with the CUDA-C frontend and runs it on the
software GPU over a fixed random 256-element fragment, checking every
cell of the resulting `16x16` tile against a Python oracle computing the
identical `row`/`col` formulas directly, requiring `max_abs_err <= 1e-9`.
Swapping the roles of `t` and `k` in either formula (a very easy mistake
given how similar the two expressions look) sends most fragment elements
to the wrong cell, which the gate catches immediately since every cell is
checked. The empty starter never writes `out` and fails outright.
