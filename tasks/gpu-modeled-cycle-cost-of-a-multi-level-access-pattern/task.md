## Context

A GPU's memory hierarchy has wildly different latencies at each level:
a register read is essentially free (~1 cycle), shared memory costs a
few dozen cycles, an L2 hit costs on the order of 200 cycles, and a
DRAM (global memory) access costs several hundred. A kernel's total
memory stall time isn't one number — it's the sum, over every access it
issues, of whichever level that particular access actually landed at.

Given a trace recording which level each access hit, and a table of
per-level latencies, the total modeled cost is just a lookup-and-sum:
$\sum_i \text{latency}[\text{level}_i]$.

## Task

Implement, in real CUDA-C:

```cuda
__global__ void access_cost(float* out, const float* level, const float* latency, int n);
```

Single-threaded (`threadIdx.x == 0` only): for `i` in `[0, n)`, look up
`lvl = (int)level[i]` (one of `0`=register, `1`=shared, `2`=L2, `3`=DRAM)
and add `latency[lvl]` to a running total; write the total to `out[0]`.

## Example

`latency = [1, 25, 200, 450]`: a trace of 2 register accesses, 1 shared,
1 L2, and 1 DRAM access costs `1+1+25+200+450 = 677` modeled cycles —
that single DRAM access alone outweighs all four of the others combined.

## What the gate checks

`max_abs_err <= 1e-6` on a fixed 40-access trace (10 register, 15
shared, 10 L2, 5 DRAM accesses, in that order) against a numpy oracle.
The trace is deliberately weighted toward the fast levels — but the 5
DRAM accesses alone (`5*450 = 2250`) still account for roughly half of
the fixed reference total (`4635`), so any wrong lookup (level indices
off by one, or the latency table read in the wrong order) shifts the sum
by a clearly detectable amount.
