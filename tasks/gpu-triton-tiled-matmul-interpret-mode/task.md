## Context

A matmul thread that reads `A` and `B` straight from global memory for
every single `k` re-fetches values that *neighboring* threads in the
same block also need — every thread in a tile's row wants the same
slice of `B`'s columns; every thread in a tile's column wants the same
slice of `A`'s rows. **Shared-memory tiling** has the whole block
*cooperate*: each thread loads exactly one element into a shared tile,
everyone waits (`__syncthreads()`), then everyone reads freely out of
that fast, on-chip tile for the chunk of the contraction it covers,
before moving to the next chunk of `K` and repeating.

The contraction dimension gets swept **one tile at a time** — load a
`TILE x TILE` slice of `A` and a `TILE x TILE` slice of `B` into shared
memory, accumulate that slice's contribution to every output element in
the block's tile, then load the next slice. Two barriers per chunk are
essential: one after loading (nobody may read the tile until everyone's
finished writing their piece of it), one after computing (nobody may
overwrite the tile with the next chunk until everyone's finished
reading the current one).

## Task

Implement, in `solve.cu`:

```cuda
__global__ void tiled_matmul(const float* A, const float* B, float* C, int M, int N, int K);
```

`TILE = 16` (fixed), one block of `256` threads covers the entire
`16x16` output (`M = N = 16`): `row = threadIdx.x / 16`,
`col = threadIdx.x % 16`. Sweep `kk` from `0` to `K` in steps of `16`:

1. `As[row*16+col] = A[row*K + kk + col]`,
   `Bs[row*16+col] = B[(kk+row)*N + col]` — each thread loads exactly
   one element of each tile.
2. `__syncthreads()`.
3. Accumulate into `acc`: for `k` from `0` to `15`,
   `acc += As[row*16+k] * Bs[k*16+col]`.
4. `__syncthreads()` before the next chunk overwrites the tiles.

After the loop, `C[row*N+col] = acc`.

## Example

`K = 32`: two chunks. Chunk 1 (`kk=0`) loads `A`'s columns `0..15` and
`B`'s rows `0..15` into shared memory, accumulates their contribution.
Chunk 2 (`kk=16`) loads `A`'s columns `16..31` and `B`'s rows `16..31`
into the *same* shared buffers (overwriting chunk 1's data, which is
fine — chunk 1's contribution is already safely folded into `acc`) and
accumulates the rest.

## What the gate checks

The grader launches `tiled_matmul` on fixed `16x32` and `32x16`
matrices (`K=32`, two tile-chunks deep) and compares the result against
an exact `A @ B` (numpy). It requires

$$
\mathrm{rel\_err} \le 10^{-9}
$$

Tiling changes *where* each value is read from during the contraction,
not the math — every element of the correct output still has to come
out exact, to ordinary floating-point precision.
