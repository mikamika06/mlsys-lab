## Context

A block-level tree reduction usually keeps going in shared memory with
`__syncthreads()` between every halving step — but once it's down to 32
live values, that's exactly one warp, and a warp is *already*
synchronous: every lane executes the same instruction at the same time,
with no barrier needed to guarantee it. Continuing to route the last 5
halvings through shared memory (with the read, the write, and the
barrier that goes with it) pays for synchronization the hardware already
gives you for free. `__shfl_down_sync` reads a value straight out of
another lane's register — no shared memory involved at all.

## Task

Implement

```cpp
__global__ void warp_final_reduce(float* out, const float* partial, int n);
```

for a single warp of 32 threads, each holding one of 32 partial sums
(`n = 32`). Sum all 32 into `out[0]` using **only** `__shfl_down_sync` —
no `__shared__` array anywhere in the kernel:

```cpp
float val = partial[tid];
val += __shfl_down_sync(0xffffffff, val, 16);
val += __shfl_down_sync(0xffffffff, val, 8);
val += __shfl_down_sync(0xffffffff, val, 4);
val += __shfl_down_sync(0xffffffff, val, 2);
val += __shfl_down_sync(0xffffffff, val, 1);
if (tid == 0) { out[0] = val; }
```

(`__shfl_down_sync` must be the whole right-hand side of its own
assignment — `val += __shfl_down_sync(...)` is fine, since the entire
right side of the `+=` is the shuffle call itself.)

## Example

At the first step (`delta=16`), lane `5` receives lane `21`'s value and
adds it in; lane `21` (and every lane `>= 16`) also does the *same*
addition with whatever lane `21+16=37` would be — out of range, so it
keeps its own value, harmlessly adding itself in a way the ladder's
later steps correct for (this is exactly how the standard shuffle-based
warp reduction ladder works: every lane runs every step, and only lane
`0`'s final value is used).

## What the gate checks

`check.py` parses `solve.cu` with the real CUDA-C frontend and runs it on
32 fixed random partial sums, comparing `out[0]` against their numpy
`sum()`. It also requires shared-memory traffic for this kernel to be
**exactly zero**:

$$
\mathrm{max\_abs\_err} \le 10^{-6}, \qquad \mathrm{smem\_waves} = 0
$$

A shared-memory tree reduction (`sdata[tid] = partial[tid];
__syncthreads(); for (stride = 16; stride > 0; stride /= 2) { if (tid <
stride) sdata[tid] += sdata[tid+stride]; __syncthreads(); }`) computes
the exact same correct sum on this fixture but measures
`smem_waves = 17` — correct isn't enough here; the point of the exercise
is eliminating that traffic entirely, not just getting the right answer
some other way.
