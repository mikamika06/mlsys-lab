## Context

This is a **real CUDA-C** task: `solve.cu` is genuine CUDA source, parsed and
executed thread-by-thread on Arena's software GPU (`arena.cuda_sim.GPU`). Every
`in[i]` / `out[i]` you write becomes a real `t.gload`/`t.gstore` call, and the
simulator reports the same thing a profiler would: how many 128-byte global
memory **transactions** your warps issued.

A warp is 32 threads executing in lockstep. When those 32 threads touch 32
consecutive 4-byte floats, the hardware **coalesces** the access into a single
128-byte transaction. When they touch scattered or strided addresses instead,
each thread (or small group) needs its own transaction — the same warp can
cost 32x more memory traffic for the exact same amount of arithmetic.

## Task

Implement the kernel

```c
__global__ void scale(float* out, const float* in, int n, float s)
```

so that `out[i] = s * in[i]` for every `i` in `[0, n)`, using one thread per
element. Compute the global index the standard way and guard against
out-of-range threads:

```c
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < n) {
    out[i] = s * in[i];
}
```

Thread `i` must read `in[i]` and write `out[i]` — not `in[some other index]`.
Any stride, permutation, or reversed indexing still computes the right
*values* (so `max_abs_err` alone wouldn't catch it) but destroys coalescing
and blows the `transactions` gate.

## Example

For `n = 256` launched as `4` blocks of `64` threads (8 warps total), a
coalesced kernel issues exactly 1 read-transaction and 1 write-transaction
per warp: `8 warps x 2 accesses = 16 transactions`. A kernel that instead
reads `in[(i * 8) % n]` computes the same values but scatters the reads
across many more 128-byte segments, multiplying the transaction count.

## What the gate checks

- **`max_abs_err`**: `max_i |out[i] - s * in[i]|` over the reference input.
  Must be `<= 1e-9` — get the math right.
- **`transactions`**: total global-memory transactions the simulator counted
  across all warps. Must be `<= 20` — only reachable with coalesced,
  one-thread-per-consecutive-element indexing (the reference kernel scores
  16; a strided or scattered kernel scores far higher and fails this gate
  even with a perfectly correct `max_abs_err`).
