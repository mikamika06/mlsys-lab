## Context

Shared memory is split into 32 banks of 4-byte words; a warp's access to
shared memory is free of conflicts only when its 32 lanes touch 32
*different* banks. A tiled transpose loads a tile row-major (fast index
= column, always conflict-free — 32 consecutive addresses trivially
spread across all 32 banks) but then has to read it back
**column-major** to produce a coalesced, transposed write to global
memory. That column-major shared read is where the trouble starts: with
an un-padded `N x N` tile, a warp reading down one column touches
addresses that are all multiples of `N` apart. When `N` is itself a
multiple of 32 (as it always is for a 32-wide tile), every one of those
addresses lands in the *same* bank — a 32-way conflict on every single
load. Padding each row by one extra element breaks that exact alignment:
`stride = N + 1` is no longer a multiple of 32, so the 32 lanes land on
32 different banks instead.

## Task

Fix the shared-memory tile transpose kernel

```c
__global__ void transpose_tile(float* out, const float* in, int n)
```

so its `__shared__` tile is declared with **one extra column of
padding** per row (`float tile[N * (N+1)]`, with `stride = n + 1` used
for *both* the store and the load into `tile`), instead of the
unpadded `float tile[N*N]` with `stride = n`. Everything else about the
kernel — thread-to-(row,col) mapping, which global addresses are read
and written — stays the same; only the shared-memory layout changes.

## Example

For a 32x32 tile (1024 threads, 32 warps): the store step, which walks a
tile row (32 consecutive shared addresses per warp), is conflict-free
either way. The load step, which walks a tile *column* per warp, is a
32-way conflict without padding (every lane's address is a multiple of
32 apart — same bank) and conflict-free with `stride = 33` (`col*33 mod
32 = col`, a bijection over the warp's 32 lanes onto the 32 banks).

## What the gate checks

- **`max_abs_err`**: the transposed output must exactly match `in.T`.
  Both the padded and unpadded layouts compute this correctly — padding
  only changes shared-memory *addressing*, never which values end up
  where.
- **`smem_waves`**: total shared-memory bank-conflict degree the
  simulator measured across all 32 warps. Must be `<= 100` — the padded
  kernel scores `64` (1 conflict-free unit per warp for each of the 2
  shared-memory steps); the unpadded kernel scores `1056` (1 for the
  conflict-free store, but 32 for the conflicted load, times 32 warps),
  over 16x higher, and fails the gate even though its output values are
  bit-for-bit identical.
