## Context

Transposing a matrix means every thread's READ address and WRITE address
can't both be coalesced at the same time — the whole point of a transpose
is to swap row-major for column-major, so whichever side is contiguous in
`in` is necessarily strided in `out`. The **naive** transpose reads
coalesced (`in[tid]`, consecutive threads → consecutive addresses) and
writes scattered (`out[c*rows + r]`, consecutive threads jump `rows`
elements apart in `out`).

## Task

Write a CUDA-C kernel:

```cpp
__global__ void naive_transpose(float* out, const float* in, int rows, int cols);
```

Transpose a `rows` x `cols` row-major matrix `in` into a `cols` x `rows`
row-major matrix `out`: `out[c][r] = in[r][c]`. One thread per input
element: `tid = blockIdx.x * blockDim.x + threadIdx.x`. Guard `tid < rows
* cols`, derive `r = tid / cols`, `c = tid % cols`, and write
`out[c * rows + r] = in[tid]`.

## Example

For an $8 \times 32$ matrix launched as 8 blocks of 32 threads (one warp
per row), each warp's READ step covers 32 consecutive elements of `in` —
1 coalesced segment. Its WRITE step scatters those same 32 values across
`out` at stride `rows = 8` (`out[0*8+r], out[1*8+r], ..., out[31*8+r]`),
spanning the ENTIRE 256-element output array — up to 32 separate 128-byte
segments touched by one warp's write. Summed across all 8 warps: `72`
total transactions (8 reads, 1 segment each, plus 8 writes, 8 segments
each: $8 \times 1 + 8 \times 8 = 72$).

## What the gate checks

The grader parses your `.cu` with the CUDA-C frontend and runs it on the
software GPU over a fixed $8 \times 32$ random fixture, requiring
`max_abs_err <= 1e-9` against numpy's `x.T` AND `transactions == 72`
against the simulator's own measurement. Getting the transposed VALUES
right through some other index derivation that doesn't match the exact
read-coalesced/write-scattered pattern of the naive transpose (say,
scattering the read instead) would still transpose correctly but land on
a different transaction count and fail the exact-match gate. The empty
starter never writes `out` and fails both.
