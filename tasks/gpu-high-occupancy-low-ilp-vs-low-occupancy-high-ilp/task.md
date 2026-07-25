## Context

A warp that issues a global-memory load stalls for the full memory
latency unless the scheduler has something else ready to run while it
waits. There are two independent sources of "something else ready":
more **resident warps** (occupancy — while warp A waits, warp B's
instructions issue instead) and more **instruction-level parallelism**
within a single warp (ILP — if a thread has already issued several
independent, not-yet-dependent memory requests, the hardware can have
all of them in flight at once instead of one at a time).

Volkov's key result is that these two sources are *interchangeable*:
what actually determines whether a memory latency gets fully hidden is
their **product** — `concurrency = warps_resident * ilp` — not
occupancy by itself. A kernel with low occupancy but high ILP per
thread can hide latency just as well as (or better than) a kernel with
maximum occupancy and no ILP, because the *total* number of independent
in-flight requests is what matters, wherever those requests come from.

## Task

Implement, in `solve.cu`:

```cuda
__global__ void latency_hiding_cycles(int warps_resident, int ilp, int compute_cycles,
                                       int mem_latency, float* out);
```

Compute `concurrency = warps_resident * ilp`. Compute
`exposed = max(0, mem_latency - concurrency)` — the portion of the
memory stall that concurrency didn't cover. Write
`out[0] = compute_cycles + exposed`.

## Example

`warps_resident=32, ilp=1, compute_cycles=200, mem_latency=256`:
`concurrency = 32`, `exposed = max(0, 256-32) = 224`,
`out[0] = 200 + 224 = 424`. `warps_resident=8, ilp=32` (same compute
and latency): `concurrency = 256`, `exposed = max(0, 256-256) = 0`,
`out[0] = 200 + 0 = 200` — a quarter of the resident warps, but 32x the
per-thread ILP, and the *lower*-occupancy kernel finishes faster.

## What the gate checks

The grader launches `latency_hiding_cycles` for 5 fixed scenarios and
compares each result against an independently computed oracle. It
requires

$$
\mathrm{exact\_match} = 1 \iff \text{every one of the 5 outputs matches the oracle}
$$

The first two scenarios are exactly Volkov's crossover: high occupancy
with `ilp=1` costs **424** cycles, while low occupancy with `ilp=32`
costs only **200** — fully hiding the same 256-cycle memory latency
with a quarter of the resident warps, because `32 * 8 = 256 \ge
256` while `1 * 32 = 32 \ll 256`.
