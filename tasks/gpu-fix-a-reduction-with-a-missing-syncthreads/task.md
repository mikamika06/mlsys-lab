## Context

A shared-memory tree reduction over `n` elements (one thread per
element) halves the number of "live" values each step:

```c
sdata[tid] = in[tid];
__syncthreads();                        // (A) every load must land first

for (stride = n/2; stride > 0; stride /= 2) {
    if (tid < stride) {
        sdata[tid] += sdata[tid + stride];
    }
    __syncthreads();                    // (B) every write must land before the next step reads it
}
```

Barrier (A) exists because step 0 reads `sdata[tid + stride]`, written
by a *different* thread's load -- without it, some threads could start
combining before others have even loaded their input. Barrier (B) is
the same idea one level up: step $k{+}1$ reads slots that step $k$'s
*other* threads just wrote, so without a barrier between steps, a
thread that finishes step $k$ early can race straight into step
$k{+}1$ and read a slot before the thread that owns it this step has
written its new value -- silently combining a stale (pre-reduction)
value instead of the correct partial sum, and corrupting the final
total.

## Task

`solve.cu` is missing the `__syncthreads()` between reduction steps.
Add it back so that every thread's write for a step is guaranteed
visible to every thread before any thread starts the next step:

```cuda
__global__ void sum_reduce(float* out, const float* in, int n) {
    int tid = threadIdx.x;
    __shared__ float sdata[8];
    sdata[tid] = in[tid];
    __syncthreads();

    int stride = n / 2;
    while (stride > 0) {
        if (tid < stride) {
            sdata[tid] = sdata[tid] + sdata[tid + stride];
        }
        // <-- the missing barrier belongs here
        stride = stride / 2;
    }

    if (tid == 0) {
        out[0] = sdata[0];
    }
}
```

## Example

For 8 fixed input values, the correct sum is a single fixed number
(whatever `sum(in[0..8))` is). Without the barrier, thread 0 -- which
participates in *every* step, since `tid < stride` is true for `tid=0`
at every stride -- finishes its own work early and reads
`sdata[tid+stride]` for the next step before the higher-indexed thread
responsible for that slot has combined its own value into it yet,
silently mixing in a pre-reduction input value instead of a partial
sum.

## What the gate checks

`check.py` runs the kernel over 8 fixed random values and checks
`max_abs_err <= 1e-9` against `numpy`'s `sum()`. This simulator has no
real hardware concurrency -- threads are stepped deterministically, so
the missing barrier doesn't produce a flaky, hardware-dependent race;
it produces one specific, exactly reproducible wrong number every time
(`0.1704...` off, for this task's fixed input, instead of `0`).
Restoring the `__syncthreads()` after the combine step is the only
change needed to make every step wait for every thread before the next
one begins.
