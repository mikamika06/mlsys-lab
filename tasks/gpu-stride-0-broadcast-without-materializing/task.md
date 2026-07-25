## Context

Adding a per-column bias to every row of a matrix is a broadcast: the
*logical* shape of the operation is `r x c` (the bias conceptually repeats
`r` times), but the bias only has `c` real numbers behind it. Materializing
that broadcast — actually allocating an `r*c`-sized copy of the bias before
adding — wastes memory and bandwidth for data that never had `r*c` degrees
of freedom to begin with. The right move is to index the *small* buffer
with a repeating pattern instead: element `i` of the flattened output
belongs to column `i % c`, so `bias[i % c]` reads the correct value every
time — a "stride 0" access across the row dimension, since moving to the
next row doesn't move the read position in `bias` at all.

## Task

Implement, in `solve.cu`, a kernel with this signature:

```cuda
__global__ void broadcast_add(float* out, const float* a, const float* bias,
                               int r, int c, int n);
```

`a` is an `r x c` row-major matrix flattened to `n = r*c` elements; `bias`
has exactly `c` real elements (never expanded to `n`). For
`i = blockIdx.x*blockDim.x + threadIdx.x` in `[0, n)`, compute
`out[i] = a[i] + bias[i % c]`.

## Example

With `r = 20, c = 8`: element `i = 3` belongs to row `0`, column `3`, so
`out[3] = a[3] + bias[3]`. Element `i = 11` belongs to row `1`, column
`3` (`11 % 8 == 3`) — the *same* column, so it reads the exact same
`bias[3]`: `out[11] = a[11] + bias[3]`, not a fresh, independent value at
some `bias[11]` that was never allocated.

## What the gate checks

`check.py` allocates `bias` with exactly `c = 8` real elements and places a
"trap" region filled with a large, recognizable value immediately after it
in memory, then parses `solve.cu` and runs `broadcast_add` on the software
GPU (`arena.cuda_sim.GPU`) with a 5-block, 32-thread launch (`160` threads,
`n = 160`). It requires `max_abs_err == 0.0` against `a + bias` broadcast
the standard (numpy) way. Indexing `bias[i]` directly — as if it had
already been expanded to `n` elements — reads correctly for row `0`
(`i < c`) and then straight into the trap for every row after it, which
the grader catches by comparing the whole output, not just the first row.
