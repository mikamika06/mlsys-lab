## Context

A classic shared-memory **tree reduction** sums `n` elements (`n` a
power of 2, one per thread) in $\log_2 n$ steps by repeatedly halving
the number of threads that still have work to do:

```c
for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
        sdata[tid] += sdata[tid + stride];
    }
    __syncthreads();
}
```

At the start of step $k$ (0-indexed), `stride` $= n / 2^{k+1}$: exactly
that many threads are still active, each combining its own partial sum
with the one `stride` slots away. Every step ends with a
`__syncthreads()` — required because step $k+1$ reads values that step
$k$'s *other* threads just wrote, and without the barrier there's no
guarantee those writes have landed yet.

Summed across all $\log_2 n$ steps, the number of *active* threads is
$n/2 + n/4 + \dots + 1 = n - 1$ — exactly the $n-1$ combines any tree
reduction over $n$ leaves must do, no matter its shape. Each active
thread performs 3 shared-memory operations per step (read `sdata[tid]`,
read `sdata[tid+stride]`, write `sdata[tid]`), on top of the $n$ initial
loads into shared memory and 1 final read by thread 0. That gives a
closed-form total shared-memory instruction count:

$$
\text{smem\_insts} = \underbrace{n}_{\text{initial loads}} + \underbrace{3(n-1)}_{\text{reduction steps}} + \underbrace{1}_{\text{final read}} = 4n - 2 .
$$

For $n = 1024$: $4 \times 1024 - 2 = 4094$.

## Task

Implement, in `solve.cu`:

```c
__global__ void block_reduce_sum(float* out, const float* in, int n);
```

for a single block of **1024 threads** reducing 1024 elements: load
`in[tid]` into `sdata[tid]`; `__syncthreads()`; then run the halving-
stride loop above (`stride = blockDim.x/2, blockDim.x/4, ..., 1`, a
`__syncthreads()` after every step); finally, thread `0` writes
`out[0] = sdata[0]`.

## Example

For `n = 1024` there are $\log_2 1024 = 10$ steps. Step $k$ has
`active_threads = 1024 / 2**(k+1)`: `512, 256, 128, 64, 32, 16, 8, 4, 2,
1`, summing to `1023 == n - 1`. Plug that schedule into the closed form
above and the reference kernel measures exactly `4094` total
shared-memory instructions on the simulator — matching the formula, not
a guessed number.

## What the gate checks

The grader parses `solve.cu` with the real CUDA-C front end
(`arena.cuda_c.CudaProgram`) and executes it thread-by-thread on the
software GPU (`arena.cuda_sim.GPU`), then requires:

- `max_abs_err <= 1e-9` — `out[0]` matches a numpy sum of the same 1024
  input values.
- `smem_insts == 4094` — the total shared-memory instruction count
  matches the derived schedule `4n - 2` for `n = 1024`.

The starter never touches shared memory, so it prints `smem_insts = 0`
and an `out[0]` that doesn't match the sum — both gates fail. A kernel
that gets the sum right by some other route (e.g. a single thread
looping over all of `in` and writing the total directly, with the other
1023 threads idle) would still fail `smem_insts`, because it never runs
the halving-active-thread schedule this task is about — the point is
the shape of the reduction, not just the final number.
