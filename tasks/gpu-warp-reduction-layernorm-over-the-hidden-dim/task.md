## Context

LayerNorm's mean and variance are reductions over the whole hidden
dimension — every element of a row contributes, so computing them
efficiently means spreading the row across many threads and combining
their partial results. **Warp shuffle** does this combination without
touching shared memory or global memory at all: `__shfl_xor_sync(mask,
var, offset)` lets every lane in a warp directly read the value another
lane is holding in its own registers, in one instruction.

A **butterfly all-reduce** applies this with a halving offset — `16, 8,
4, 2, 1` for a 32-lane warp — where at each step, every lane adds in
the value held by the lane `offset` away (found by XOR-ing its own lane
id with `offset`). After all 5 steps, **every** lane ends up holding the
exact same total — the whole warp reduces to a single sum without a
separate "broadcast the answer back out" step, because XOR-based
pairing is symmetric: if lane A reads from lane B at some step, lane B
also reads from lane A at that same step.

## Task

Implement, in `solve.cu`:

```cuda
__global__ void warp_layernorm(const float* x, const float* gamma, const float* beta,
                                float* y, int rows, int D, float eps);
```

One warp (32 threads) per row: `row = global_tid / 32`,
`lane = global_tid % 32`, `D = 128` (4 elements per lane, strided by
32: `x[row*D + lane + c]` for `c = 0, 32, 64, 96`).

1. Each lane accumulates `sum` and `sumsq` over its own 4 elements.
2. All-reduce both via butterfly XOR-shuffle, offsets `16, 8, 4, 2, 1`
   in that order: `float tmp = __shfl_xor_sync(0xffffffff, sum, offset);`
   (the shuffle call must be the *entire* right-hand side of its own
   statement) then `sum = sum + tmp;` as a separate line. Same shape
   for `sumsq`.
3. `mean = sum/D`, `var = sumsq/D - mean*mean`,
   `invstd = 1/sqrt(var+eps)` — every lane now has the row's true
   mean and variance.
4. Each lane writes its own 4 elements:
   `y[row*D+d] = (x[row*D+d]-mean)*invstd*gamma[d] + beta[d]`.

## Example

`D=128`: lane `5` sums `x[5], x[37], x[69], x[101]`. After the
butterfly reduction, lane `5` (and every other lane in the warp) holds
the sum of all 128 elements of that row — not just its own 4.

## What the gate checks

The grader launches `warp_layernorm` on a fixed `4 x 128` input (4
warps, one per row) with random `gamma`/`beta`, and compares the output
against numpy's own row-wise LayerNorm. It requires

$$
\mathrm{max\_abs\_err} \le 10^{-5}
$$

The reduction has to be a genuine warp-wide all-reduce — if a lane
normalizes using only its own 4-element partial sum instead of the
whole row's total, every output is wrong by a large, consistent amount.
