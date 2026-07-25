## Context

Two classic parallel scan algorithms compute the same result by very
different amounts of total work:

- **Hillis-Steele**: at every one of `log2(n)` steps, ALL `n` threads are
  active (each either adds two values or just copies forward) — `O(n log n)`
  total operations, more than the `n - 1` additions a sequential scan
  needs.
- **Blelloch (work-efficient)**: up-sweep then down-sweep, and at step
  `d` only `d` threads are active (`d` shrinking to `1` on the way up,
  growing back to `n/2` on the way down) — `O(n)` total additions.

The asymptotics say Blelloch always wins. Measured reality is more
textured: every thread in a block — active or not — still pays for its
own index bookkeeping (recomputing `offset`, the loop counter, `ai`/`bi`)
every step, on real SIMT hardware exactly as in this simulator. At small
`n`, that fixed per-thread overhead can outweigh Blelloch's smaller
active-thread-count; the asymptotic win only shows up once `n` is large
enough for it to dominate. This task measures where that crossover
already shows up.

## Task

Implement both:

```cuda
__global__ void hillis_steele_scan(float* out, const float* in, int n);
__global__ void blelloch_scan(float* out, const float* in, int n);
```

Both compute the INCLUSIVE scan of `n = 256` elements, one block of `256`
threads.

**`hillis_steele_scan`**: double-buffer in `__shared__ float temp[512]`
(`temp[0..n)` and `temp[n..2n)`) to avoid a read/write race within a
step. Load `temp[tid] = in[tid]`, `__syncthreads()`. Then for
`offset = 1, 2, 4, ...` while `offset < n`: swap the buffer pair
(`pout`/`pin` flip between `0`/`1`), `__syncthreads()` first each step,
then every thread with `tid >= offset` writes
`temp[pout*n+tid] = temp[pin*n+tid] + temp[pin*n+tid-offset]`; every
other thread just copies `temp[pout*n+tid] = temp[pin*n+tid]`.
`__syncthreads()`, double `offset`. Finally `out[tid] = temp[pout*n+tid]`.

**`blelloch_scan`**: the up-sweep/down-sweep algorithm (same as
`gpu-blelloch-work-efficient-scan-up-sweep-down-sweep`, `ai =
offset*(2*tid+1)-1`, `bi = offset*(2*tid+2)-1`), but INCLUSIVE: after the
down-sweep produces the exclusive scan in `temp[tid]`, write
`out[tid] = temp[tid] + in[tid]`.

## Example

`in = [3, 1, 4, 1]` (`n = 4`, inclusive): both algorithms must produce
`out = [3, 4, 8, 9]` — `out[i]` is the sum of everything up to and
including index `i`.

## What the gate checks

`check.py` seeds a fixed random 256-element input, launches both kernels
on independent GPUs, and checks their outputs against a numpy
`np.cumsum` oracle. It also reads each launch's `cycles`, `transactions`,
and `smem_waves` and backs out the pure arithmetic-op count:
`ops = cycles - transactions*200 - smem_waves*20` (every `+ - * / %` the
kernel evaluates, index math included — exactly what a real ALU-op
counter would show, not just the scan's own additions). It requires

$$
\mathrm{max\_abs\_err} \le 10^{-6}, \quad \mathrm{ops\_hillis} = 708, \quad \mathrm{ops\_blelloch} = 489
$$

At `n = 256`, Blelloch's real measured op count (`489`) is already
`31%` below Hillis-Steele's (`708`) — the asymptotic advantage has
started to show. (For the curious: at `n = 32` it's the other way
around, `113` vs `59` — Blelloch's per-active-thread index arithmetic
costs more than Hillis-Steele saves by having fewer active threads, until
`n` grows enough for the step-count savings to win out.)
