## Context

Summing `blockDim.x` values into one number with a **tree reduction**
means $\log_2(\text{blockDim.x})$ steps, each step adding half as many
values as the last, all staged through `__shared__` memory. HOW you
index into shared memory at each step matters enormously, even though
every valid indexing scheme computes the exact same sum.

**Interleaved addressing** (the naive version) picks active threads with
a modulo test, `tid % (2*stride) == 0`, and grows `stride` from `1`
upward. **Sequential addressing** instead picks a *contiguous* range of
active threads, `tid < stride`, and *shrinks* `stride` from
`blockDim.x/2` downward. Both are correct. Sequential addressing is the
one real CUDA code uses, because at every step `tid` and `tid + stride`
land in *different* shared-memory banks for every active thread
simultaneously — zero bank conflicts — while interleaved addressing
repeatedly funnels many threads' accesses through the same handful of
banks.

## Task

Implement

```cpp
__global__ void block_reduce_sum(float* out, const float* in, int n);
```

launched as a single block of 256 threads reducing 256 elements:

1. `sdata[tid] = in[tid];` then `__syncthreads();`
2. For `stride = blockDim.x/2; stride > 0; stride /= 2`: if `tid < stride`,
   `sdata[tid] += sdata[tid + stride];` then `__syncthreads();` (every
   iteration, whether or not this thread was active — the barrier itself
   is not conditional).
3. Thread `0` writes `out[0] = sdata[0];`.

## Example

`blockDim.x = 256`: step 1 has `stride=128`, threads `0..127` each add
`sdata[tid+128]` into `sdata[tid]` — 128 pairwise sums computed in
parallel, all bank-conflict-free. Step 2, `stride=64`, works on the
result; after 8 steps (`128, 64, 32, 16, 8, 4, 2, 1`), `sdata[0]` holds
the sum of all 256 original values.

## What the gate checks

`check.py` parses `solve.cu` with the real CUDA-C frontend and runs it on
the software GPU over a fixed 256-element random input, launched as one
block of 256 threads. It requires

$$
\mathrm{max\_abs\_err} \le 10^{-9} \quad \text{(the sum itself, out[0]
matches numpy's own sum)}
$$
$$
\mathrm{smem\_waves} \le 50
$$

`smem_waves` counts shared-memory bank-conflict traffic across the whole
kernel. On this fixture the reference's sequential-addressing reduction
measures `smem_waves=45` (essentially conflict-free — one wave per
active thread per step, no extra serialization); an otherwise-CORRECT
interleaved/modulo tree reduction (`idx = 2*stride*tid`, `stride` growing
from `1`) computes the exact same sum but measures `smem_waves=150` —
over 3x more bank-conflict traffic — because it repeatedly funnels many
threads through the same banks instead of spreading them out. Getting the
*sum* right is not enough; the shared-memory access pattern has to be
right too.
