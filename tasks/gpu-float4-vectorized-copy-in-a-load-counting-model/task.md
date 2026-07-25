## Context

A `float4` load reads 4 consecutive floats in a single instruction instead
of 4 separate ones — free throughput whenever data is 16-byte aligned and
you need 4-at-a-time anyway. The catch is the tail: if `n` isn't a
multiple of 4, the last `n % 4` elements don't fill a whole `float4`, and
have to fall back to ordinary scalar loads, one instruction each. The
total number of load instructions a vectorized copy issues is therefore
$\lfloor n/4 \rfloor + (n \bmod 4)$ — one per group of 4, plus one per
leftover scalar — never $n$.

This CUDA-C subset has no real `float4` type, so this task models the
instruction count explicitly instead of relying on the simulator to infer
it: each thread that handles a group of 4 reports exactly one "load"; each
thread that handles one leftover scalar also reports one; every other
thread reports zero.

## Task

Implement, in `solve.cu`, a kernel with this signature:

```cuda
__global__ void vectorized_copy_load_count(float* out, float* load_flag,
                                            const float* in, int n);
```

`num_groups = n/4`, `tail = n%4` (integer division). For thread
`t = threadIdx.x`:

- if `t < num_groups`: copy the 4 elements `in[4t..4t+3]` to
  `out[4t..4t+3]`, and set `load_flag[t] = 1.0f` (one modeled load
  instruction for the whole group).
- else if `t < num_groups + tail`: copy the single element
  `in[num_groups*4 + (t - num_groups)]` to the same index in `out`, and
  set `load_flag[t] = 1.0f`.
- else: set `load_flag[t] = 0.0f` (this thread does nothing).

## Example

For `n = 50`: `num_groups = 12`, `tail = 2`, so threads `0..11` each copy a
group of 4 (covering elements `0..47`), threads `12..13` each copy one
leftover element (`48`, `49`), and threads `14..31` (the launch has 32
threads total) report `0`. Summed over all 32 threads,
`load_flag` totals `12 + 2 = 14`, matching
$\lfloor 50/4 \rfloor + (50 \bmod 4) = 12 + 2 = 14$ exactly — far fewer
than one load per element (`50`).

## What the gate checks

`check.py` parses `solve.cu` and runs `vectorized_copy_load_count` on the
software GPU (`arena.cuda_sim.GPU`) with a 1-block, 32-thread launch over
`n = 50`. It requires `max_abs_err == 0.0` (`out` must be a byte-exact copy
of `in`) **and** `load_count_err == 0.0` (the sum of `load_flag` across all
32 threads must equal `floor(n/4) + n%4`, computed independently in
Python). Reporting `load_flag[t] = 1.0f` for a group thread but forgetting
the tail threads' flags gets the copy right (`max_abs_err = 0.0`) while
under-reporting the load count by exactly `tail`.
