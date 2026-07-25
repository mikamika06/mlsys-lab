## Context

A warp is already synchronous — every lane executes the same instruction
at the same time — so reducing 32 values doesn't need shared memory or
`__syncthreads()` at all. `__shfl_down_sync(mask, val, delta)` lets every
lane directly read another lane's register: lane `L` gets lane `L+delta`'s
current `val`. Halving `delta` each step (`16, 8, 4, 2, 1`) sums the
whole warp in exactly 5 steps — the same log-step doubling structure as a
shared-memory tree reduction, but register-to-register, no memory traffic
at all.

Unlike a prefix-sum (`__shfl_up_sync`), the reduction doesn't need a
per-lane guard: whichever lane's `lane+delta` partner falls outside the
warp, `__shfl_down_sync` just returns that lane's own value back — adding
it in doubles a partial sum that's never read again. Only lane 0's value,
after all 5 steps, matters: it has folded in every other lane's
contribution by then.

## Task

Implement, in real CUDA-C:

```cuda
__global__ void warp_reduce_sum(float* out, const float* in, int n);
```

`lane = threadIdx.x % 32`, `warp = threadIdx.x / 32`, `val = in[threadIdx.x]`.
Run 5 steps, `delta = 16, 8, 4, 2, 1`: `float got = __shfl_down_sync(0xffffffff,
val, delta); val = val + got;`. Then, only from `lane == 0`, write
`out[warp] = val`.

## Example

32 lanes each holding `1.0`: after `delta=16`, every lane's `val` is
`2.0` (folded in one neighbor 16 away). After `delta=8`, `4.0`. ... after
all 5 steps, lane 0 holds `32.0` — the full warp sum — even though lane 0
never directly talked to lane 31.

## What the gate checks

`max_abs_err <= 1e-9` on two warps' worth (64 elements, one block) of
fixed random input, against a numpy oracle summing each 32-element half.
Using `__shfl_up_sync` instead of `_down_`, guarding the adds (unlike the
scan, this ladder needs none — a guard here silently drops valid partial
sums), or writing `out[]` from any lane other than `lane == 0`, all
produce the wrong per-warp totals.
