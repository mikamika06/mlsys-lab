## Context

FlashAttention fuses the whole attention computation into one kernel by
never materializing the full $N \times N$ score matrix: it tiles the
sequence into blocks and streams $Q$, $K$, $V$ tiles through fast
on-chip shared memory (SRAM) instead of round-tripping through slow
global memory. That only works if a block's tiles actually **fit** in
SRAM — a fixed, small budget (tens of KB) shared by the whole
thread block.

With square tiling — block row count $B_r$ equal to block column count
$B_c$, both equal to some $T$ — a block needs four $T \times d$ tiles
(where $d$ is the head dimension): $Q$ and the output accumulator $O$
are each $T \times d$, and so are $K$ and $V$. Four `float32` tiles of
$T \times d$ elements cost $4 \times T \times d \times 4$ bytes total.
Solving $4Td \cdot 4 \le \text{sram\_bytes}$ for $T$ gives the largest
block size that fits — rounded down to a power of two, since real
kernels want block sizes that divide evenly into warps.

## Task

Implement, in `solve.cu`:

```cuda
__global__ void derive_tile_size(int sram_bytes, int head_dim, float* out);
```

Compute `raw = floor(sram_bytes / (16 * head_dim))` (the largest $T$
that fits, before rounding to a power of two — `16` because
$4 \text{ tiles} \times 4 \text{ bytes per float} = 16$). If `raw < 1`,
write `out[0] = 0` (nothing fits). Otherwise, round `raw` down to the
nearest power of two: `e = floor(log2(raw))`, `out[0] = 2^e`.

## Example

`sram_bytes = 49152` (48 KB), `head_dim = 64`: `raw = floor(49152 / (16
* 64)) = floor(48.0) = 48`. The nearest power of two at or below `48`
is `32` (`64` would need `4 * 64 * 64 * 4 = 65536` bytes, over budget).
`out[0] = 32.0`.

## What the gate checks

The grader launches `derive_tile_size` once per `(sram_bytes,
head_dim)` pair over 5 fixed scenarios, comparing each result against
an oracle computed independently in Python with the same formula. It
requires

$$
\mathrm{exact\_match} = 1 \iff \text{every one of the 5 outputs matches the oracle exactly}
$$

The 5 scenarios span a spread of budgets and head dimensions on
purpose — `(49152, 64) -> 32`, `(98304, 64) -> 64`, `(49152, 128) ->
16`, `(16384, 32) -> 32`, `(65536, 16) -> 256` — so a formula that's
right for one case (e.g. one that forgets to round to a power of two,
or off-by-one's the `raw` floor) fails on at least one of the others.
