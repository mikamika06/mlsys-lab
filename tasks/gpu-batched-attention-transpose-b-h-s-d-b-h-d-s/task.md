## Context

Attention implementations constantly reshape between a "sequence-major"
layout $(B, H, S, D)$ (convenient for the QK^T matmul) and a
"head-dim-major" layout $(B, H, D, S)$ (convenient for other stages).
For batch $b$, head $h$, position $s$, feature $d$, the flat offsets in
each row-major layout are

$$
\mathrm{idx_{in}}(b,h,s,d) = ((b H + h) S + s) D + d, \qquad
\mathrm{idx_{out}}(b,h,d,s) = ((b H + h) D + d) S + s.
$$

A GPU global-memory transaction moves a whole 128-byte segment
regardless of how many of its bytes a warp's 32 threads actually
wanted. If consecutive threads in a warp touch consecutive addresses,
one transaction serves the whole warp (**coalesced**); if they touch
addresses `S` elements apart instead, the warp's accesses scatter across
many segments (**strided**, many transactions for the same 32 useful
values).

A transpose is exactly the case where you cannot have it both ways for
free: the input's fastest-varying axis ($d$) becomes the output's
*second*-fastest axis, so a thread layout that reads `in` coalesced
necessarily writes `out` strided, and vice versa. The standard fix is to
**bounce through `__shared__` memory**: read a tile from `in` with one
coalesced access pattern, `__syncthreads()`, then write it to `out` with
a *different* (also coalesced) access pattern -- the actual
corner-turning happens entirely on-chip, where a strided access doesn't
cost extra global-memory transactions.

## Task

Implement:

```cuda
__global__ void transpose_bhsd(float* out, const float* in, int B, int H, int S, int D);
```

`in` holds a $(B,H,S,D)$ tensor, `out` a $(B,H,D,S)$ tensor, both
row-major and flattened. Launch one block per $(b,h)$ slice
(`blockIdx.x` ranges over `B*H`), with `S*D` threads per block (one per
element of that slice's $S \times D$ tile):

1. Each thread loads one element of the tile from `in` into
   `__shared__ float tile[256]` (sized for this task's fixed
   `S = D = 16`), indexed so that consecutive `threadIdx.x` read
   consecutive `in` addresses.
2. `__syncthreads()`.
3. Each thread writes one element of `out`, indexed so that consecutive
   `threadIdx.x` write consecutive `out` addresses -- reading the
   *transposed* position back out of `tile`.

## Example

For `threadIdx.x = tid` within a block, `in_row = tid / D`,
`in_col = tid % D` reads `in[block_base + tid]` into
`tile[in_row * D + in_col]` (coalesced: `tid` increasing by 1 moves
`in_col`, the fastest input axis, by 1). Then `out_s = tid % S`,
`out_d = tid / S` writes `out[block_base_out + tid]` from
`tile[out_s * D + out_d]` (also coalesced: `tid` increasing by 1 moves
`out_s`, the fastest output axis, by 1) -- the *value* placed at each
`tid` is the correctly transposed one, because the address into `tile`
swaps which coordinate is fast-varying instead of the address into
global memory.

## What the gate checks

`check.py` runs the kernel over a fixed `(B,H,S,D) = (2,2,16,16)`
random tensor, one block per of the `4` `(b,h)` slices, `256` threads
per block. It checks `max_abs_err <= 1e-9` against
`numpy`'s `x.reshape(B,H,S,D).transpose(0,1,3,2)`, and
`transactions <= 100` from the simulator's coalescing model. The
reference's tiled approach measures `64` transactions. A per-thread
kernel that skips the `__shared__` bounce and writes straight from
`in[gid]` to the transposed `out` index is still numerically correct
(passes `max_abs_err`) but writes with a `D`-element stride at this
tile size, measuring `288` transactions -- well over the gate, because
correctness alone doesn't prove the access pattern is efficient.
