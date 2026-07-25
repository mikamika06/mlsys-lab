## Context

Transposing a matrix has two structurally different strategies:

- **Out-of-place**: every thread reads one element of the source and
  writes it to the mirrored position in a *second*, freshly allocated
  buffer. Simple, works for any (even non-square) shape, but needs a full
  extra copy of the data resident in memory at once.
- **In-place**: only defined for square matrices, and only sensible with
  *half* the threads doing anything. If every thread swapped
  `A[row][col]` with `A[col][row]` unconditionally, each pair would be
  swapped once by the `(row, col)` thread and then swapped right back by
  the `(col, row)` thread — a no-op. Only the upper-triangle threads
  (`row < col`) may act; each one swaps its element with its mirror, which
  correctly relocates both halves of the pair using zero extra memory.

The trade isn't free: the in-place kernel's signature only has room for one
pointer, so "extra buffer" isn't a number to measure at runtime — it's
`0` by construction, structurally impossible to violate. What it costs
instead is applicability (square shapes only) and a swap-partner access
pattern (`A[col*n+row]`, a transposed *index* of a transposed *index*) that
doesn't coalesce as cleanly as out-of-place's direct read-then-write. On
this task's fixture, measured directly from the simulator, in-place
actually issues *more* global-memory transactions (`88`) than
out-of-place (`72`), even though only half its threads do any work at all
— a reminder that "moves less data" and "issues fewer coalesced
transactions" are not the same claim, and the real, unconditional win of
in-place transpose is memory *capacity*, not transaction count.

## Task

Implement, in `solve.cu`, TWO kernels (this CUDA-C subset allows several
distinctly-named `__global__` functions in one file, just not two sharing a
name):

```cuda
__global__ void transpose_in_place(float* A, int n);
__global__ void transpose_out_of_place(float* out, const float* in, int n);
```

Both launch with `n*n` threads, `tid -> (row = tid/n, col = tid%n)`,
`n = 16`.

- `transpose_in_place`: if `row < col`, swap `A[row*n+col]` and
  `A[col*n+row]`. Every other thread (`row >= col`) does nothing.
- `transpose_out_of_place`: every thread writes
  `out[col*n+row] = in[row*n+col]`, unconditionally; `in` is never
  modified.

## Example

Both kernels transpose the exact same random `16x16` matrix, with
`max_abs_err = 0.0` for a correct implementation of either (pure data
movement, no arithmetic, so there's nothing to round). The starter's TODOs
leave both bodies empty: `in_place` never swaps anything (`A` stays as its
original, un-transposed self, `max_abs_err ~ 1.85`), and `out_of_place`
never writes anything (`out` stays at its pre-launch fill of `0.0`,
`max_abs_err ~ 0.99`).

## What the gate checks

`check.py` parses `solve.cu`, then runs **both** kernels — `transpose_in_place`
on a single 16x16 buffer, `transpose_out_of_place` on a source buffer plus a
separate destination buffer — and requires `in_place_max_abs_err == 0.0`
**and** `out_of_place_max_abs_err == 0.0`. Swapping unconditionally for
every thread (no `row < col` guard) passes `out_of_place` but fails
`in_place`: every pair gets swapped twice and lands back where it started,
`max_abs_err = 0.0`'s opposite — the matrix comes out completely
unchanged, not transposed.
