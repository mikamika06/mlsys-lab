## Context

Floating-point addition is not associative: summing the same values in a
different order can produce a different rounded result. This normally
shows up as noise many orders of magnitude below anything you'd notice —
until the running total's magnitude grows far past the magnitude of the
values still being folded into it. At that point, a term's contribution can
round away to nothing: if the accumulator's ULP (the gap between adjacent
representable values at its magnitude) is larger than the term being added,
`accumulator + term` rounds right back to `accumulator`.

A GPU block-wide reduction is a live instance of this. Summing `in[0..n)`
one thread at a time into a single running total is a strictly *sequential*
accumulation order — every term, however small, gets folded directly into
whatever the total has already grown to. A balanced shared-memory **tree**
reduction instead combines nearby, similarly-scaled partial sums with each
other first, and only merges the largest values in at the very end, when
the combining value is itself already comparably large. Same set of inputs,
same true mathematical answer, very different floating-point paths to it —
and very different amounts of precision lost along the way.

## Task

Implement, in `solve.cu`, a kernel with this signature:

```cuda
__global__ void reduce_sum(float* out, const float* in, int n);
```

One block, `n == blockDim.x` threads, one input element per thread. Load
`in[tid]` into `__shared__` storage, then run a balanced tree reduction:
for `s = blockDim.x/2, blockDim.x/4, ..., 1`, every thread with
`tid < s` adds `sdata[tid + s]` into `sdata[tid]` (sequential addressing —
this is also bank-conflict-free), with a `__syncthreads()` barrier between
rounds so every round's reads see every previous round's writes. After the
last round, thread 0 writes `sdata[0]` to `out[0]`.

## Example

The grader's fixture is one huge value ($2^{54}$, whose ULP there is $4$)
followed by 127 values drawn from $[0.1, 2.0)$ — individually far smaller
than that ULP. A **balanced tree reduction** first combines pairs, then
pairs of pairs, and so on: the 127 small values merge with each other for
several rounds, growing into partial sums well past the huge value's ULP,
*before* that lineage ever gets added to the huge value — so on this
fixture the reference implementation loses **nothing**: its result matches
`math.fsum` (Python's correctly-rounded exact summation) exactly,
`max_abs_err = 0.0`.

Contrast that with folding the terms into a single running total **one at a
time**, in index order (the huge value first, then each small value added
individually): every single small addition is swallowed on its own before
it ever gets a chance to accumulate with its neighbors, and the result comes
out `140.0` short of the exact sum — losing essentially the entire
contribution of all 127 small values.

## What the gate checks

`check.py` builds the fixture above, computes the exact reference sum with
`math.fsum`, parses `solve.cu`, and runs `reduce_sum` on the software GPU
(`arena.cuda_sim.GPU`) with a 1-block, 128-thread launch. It requires
`max_abs_err <= 1e-6` (the reference tree reduction achieves exactly `0.0`
on this fixture; any implementation that folds the huge value in early and
keeps accumulating small terms into it — including summing everything in a
single thread's serial loop — comes out around `140.0` short) and
`smem_waves <= 40` (the reference's sequential-addressing pattern measures
`29`; an alternative reduction using the classic *interleaved*-addressing
pattern — `index = 2*s*tid`, `sdata[index] += sdata[index + s]` — computes
the exact same numeric answer, `max_abs_err = 0.0`, but its scattered
per-round active-thread pattern measures `smem_waves = 74`, comfortably
over the gate).
