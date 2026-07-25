## Context

Transposing a matrix moves every element to a different position:
`B[j][i] = A[i][j]`. That single fact means a thread's most natural
mapping for reading `A` and its most natural mapping for writing `B` pull
in opposite directions. A warp coalesces into few transactions only when
consecutive lanes touch consecutive addresses — read `A` row-major with
consecutive lanes sweeping across a row (coalesced), and those same lanes
now want to write `B[col][row]` for consecutive `col` values, which lands
`n` elements apart in `B`'s row-major storage: one 128-byte transaction
per lane instead of one per warp.

The standard fix stages through `__shared__` memory instead of forcing one
thread mapping to serve both directions. Read `A` with the mapping that
coalesces the read, store it into a shared tile; after a barrier, read the
*transposed* position back out of that (fast, on-chip) tile, and write `B`
with the mapping that coalesces the write. Both slow, off-chip directions
end up coalesced; only the on-chip shared-memory traffic in between pays
for the mismatch.

## Task

Implement, in `solve.cu`, a kernel with this signature:

```cuda
__global__ void tiled_transpose(float* B, const float* A, int n);
```

`A` and `B` are both `n x n`, row-major. Compute `B = A^T`. One block,
`n*n` threads. Stage through `__shared__ float tile[256]`: read
`A[row*n+col]` (with `col` varying fastest across the warp) into
`tile[row*n+col]`, `__syncthreads()`, then write
`B[row2*n+col2] = tile[col2*n+row2]` (with `col2` varying fastest) — both
global-memory directions coalesced, only the shared read in between
(`col2*n+row2`, stride-n) pays the mismatch cost.

## Example

For `n = 16` (256 threads, 8 warps), the grader reports:

```
staged (reference): transactions = 16
naive (write straight from the read mapping, no staging): transactions = 72
```

Both produce a byte-exact transpose (`max_abs_err = 0.0` either way — pure
data movement) — but the naive version, which reads `A` coalesced and
writes `B[col*n+row]` straight from that same mapping, needs 4.5x more
global-memory transactions for the exact same result.

## What the gate checks

`check.py` builds a random `16x16` matrix, parses `solve.cu`, and runs
`tiled_transpose` on the software GPU (`arena.cuda_sim.GPU`) with a
1-block, 256-thread launch. It requires `max_abs_err <= 1e-9` **and**
`transactions <= 20`. A kernel that transposes correctly but writes `B`
directly from the read-coalescing thread mapping (no `__shared__` staging
at all) passes correctness outright — and still fails the gate on
`transactions = 72`.
