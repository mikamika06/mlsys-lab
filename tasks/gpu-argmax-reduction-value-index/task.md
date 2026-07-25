## Context

A max-reduction across a block collapses $n$ values down to one via a
shared-memory tree: pair up neighbors, keep the larger, halve the active
thread count, repeat $\log_2 n$ times. Tracking *which* element won —
argmax, not just max — means carrying a second shared array of indices
alongside the values and updating both together on every merge.

Ties need a rule, or two otherwise-correct implementations can legitimately
disagree. This task fixes it: **the lowest original index wins**. That
rule is only automatically satisfied if the reduction tree is structured
so that, at every merge, the *left* operand's provenance (the range of
original indices it could have come from) is entirely below the *right*
operand's — grow the active stride from $1$ upward (`stride`, `2*stride`,
`4*stride`, ...; only threads with `tid % (2*stride) == 0` stay active),
and thread `tid`'s current winner always represents indices strictly below
thread `tid + stride`'s. Keeping the left side on an exact tie (`>`, never
`>=`) then always keeps the lower index, at every level, automatically.

## Task

Write a CUDA-C kernel:

```cpp
__global__ void argmax_reduce(const float* in, float* out, int n);
```

Launched as a single block of `n = 32` threads. Reduce `in[0..n)` to its
maximum value and the (lowest, on a tie) index where it occurs, writing
`out[0] = max value`, `out[1] = argmax index` (as a float).

1. Each thread `tid` seeds `sval[tid] = in[tid]`, `sidx[tid] = tid` into
   `__shared__` arrays, then `__syncthreads()`.
2. `stride` starts at `1` and doubles each round while `stride < n`. Each
   round, only threads with `tid % (2 * stride) == 0` compare
   `sval[tid + stride]` against `sval[tid]`; if the right side is
   *strictly* greater, both `sval[tid]` and `sidx[tid]` are overwritten
   from `tid + stride`. `__syncthreads()` after every round.
3. Thread `0` writes `out[0] = sval[0]`, `out[1] = sidx[0]`.

## Example

On a 32-element fixture with two elements deliberately tied at the
maximum value (indices 3 and 27 both set to `5.0`):

```
out[0] = 5.0   // the max value
out[1] = 3.0   // index 3, not 27 -- the lower of the two tied indices
```

`numpy.argmax` on the same array already returns the first (lowest)
matching index for exactly this reason, and is the oracle the grader
checks against.

## What the gate checks

The grader parses your `.cu` with the CUDA-C frontend and runs it
thread-by-thread on the software GPU (`arena.cuda_sim.GPU`) over a 32-value
fixture that includes a forced tie, then checks `out[0]` against the true
max (`max_abs_err <= 1e-9`) and `out[1]` against the lowest tied index
(`index_exact == 1.0`). Reducing the stride from `n/2` down to `1` instead
(comparing `tid` against `tid + stride` with the ACTIVE set being the
lower half at every level) also finds the correct max *value* — but it
groups original indices by an interleaved, non-contiguous pattern, so
"keep the left operand on a tie" no longer means "keep the lower index":
it can report a higher-indexed tied winner instead, and the gate catches
it (`index_exact` fails even though `max_abs_err` alone might not). The
empty starter leaves `out` at its `-1.0` sentinel and fails both gates.
