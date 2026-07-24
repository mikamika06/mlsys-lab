## Context

Shared memory is split into 32 banks of 4-byte words; `bank = word_index % 32`. A warp can service one word per bank per cycle, so 32 lanes hitting 32 different banks costs one wave, and 32 lanes hitting the *same* bank costs 32.

A tiled transpose writes the tile row-wise and reads it column-wise. With a 32-wide tile, column `c` lives at words `c, c+32, c+64, ...` — all in bank `c % 32`. Every column read is a 32-way conflict.

Padding the row stride to 33 shifts each row by one bank, so a column walks all 32 banks instead of one. One extra word per row, 32x fewer waves.

This simulator's CUDA-C frontend only carries the `.x` component through `threadIdx`/`blockIdx`/`blockDim` (`.y` always reads as `0`), so the kernel is launched as a single 1D block of `1024` (`32*32`) threads, and `threadIdx.x` is manually split into `(row, col)` — exactly the `(linear_tid / 32, linear_tid % 32)` split real hardware uses internally to assign warps for a 32x32 2D block.

## Task

`solve.cu` contains a *broken* (unpadded) `transpose_tile`. Fix it: pad the shared tile's row stride from `32` to `33` words.

```cuda
__global__ void transpose_tile(float* out, const float* in, int n) {
    __shared__ float tile[1056];   // 32 * 33
    int tid = threadIdx.x;
    int row = tid / 32;
    int col = tid % 32;
    tile[row * 33 + col] = in[row * n + col];
    __syncthreads();
    out[row * n + col] = tile[col * 33 + row];
}
```

Do not change the transpose logic itself (which element goes where) — only the shared-memory stride.

## Example

For a `32x32` tile, the conflict-free (padded) version measures `64` shared-memory waves; the unpadded one measures `1056` — same correct output, `16.5x` the shared-memory traffic.

## What the gate checks

`check.py` compiles the candidate `.cu`, runs it on the software GPU, and checks two things: `max_abs_err <= 1e-9` against `numpy`'s `A.T`, and `smem_wave_ratio <= 1.05`, where the denominator is a conflict-free wave count the grader measures itself by running its own padded kernel through the same simulator — never hardcoded. An unpadded `32`-wide stride lands at a ratio around `16.5` and fails, even though its output is numerically correct.
