## Context

A plain scan runs across the whole array. A **segmented scan** instead
restarts at marked boundaries: given `head_flag[i] == 1` at the first
element of each segment, `out[i]` should be the sum of `in[k]` from
*that segment's own head* through `i` — never reaching back across a
segment boundary into a previous segment's values.

The intra-warp shuffle scan already moves a running `val` up the lanes.
Segmenting it means shuffling a second quantity, `flag`, alongside it:
`flag` starts as this lane's own `head_flag`, and at every step it
propagates forward via `max` (once any lane in a growing window has
crossed a head, every lane further right needs to know that). `val`
only absorbs a shuffled-in contribution while `flag` is still `0` — the
moment a lane's `flag` becomes `1` (it has itself crossed its own
segment's head), `val` stops absorbing anything from the left, forever,
even though the ladder keeps running.

## Task

Implement

```cpp
__global__ void segmented_scan(float* out, const float* in, const float* head_flag, int n);
```

for one warp (`n = 32`). `lane = threadIdx.x % 32`. Initialize
`val = in[tid]`, `flag = head_flag[tid]`. For `delta` in
`1, 2, 4, 8, 16` (each step, in order):

1. `val_up = __shfl_up_sync(0xffffffff, val, delta);`
2. `flag_up = __shfl_up_sync(0xffffffff, flag, delta);`
3. If `lane >= delta`: if `flag == 0`, `val = val + val_up;` — **then**
   `flag = fmaxf(flag, flag_up);` (update `flag` even on steps where
   `val` wasn't merged, and use the value `flag` had *before* this
   step's update to decide whether to merge).

Finally `out[tid] = val;`.

## Example

Segment heads at lanes `0` and `5` (`head_flag = [1,0,0,0,0,1,0,...]`):
lane `5`'s own `flag` is already `1` from the start, so it never merges
anything from lanes `0..4` at any step — `out[5] = in[5]` exactly. Lane
`7` merges lane `6` (`delta=1`, `flag` still `0`), then at `delta=2`
tries to merge lane `5`'s POST-step-1 value — but by then lane `7`'s own
`flag` has already become `1` (propagated from lane `6`, which propagated
from lane `5`), so the `delta=2` merge is skipped: `out[7] = in[7] +
in[6]`, not `in[7]+in[6]+in[5]`.

## What the gate checks

`check.py` parses `solve.cu` with the real CUDA-C frontend and runs it on
32 fixed elements with 8 segments (heads at lanes 0, 5, 6, 13, 20, 21,
22, 30), comparing the output against a reference computed by grouping
elements by segment and running `numpy.cumsum` within each group. It
requires

$$
\mathrm{max\_abs\_err} \le 10^{-6}
$$

Running the plain (unsegmented) shuffle scan — ignoring `head_flag`
entirely — measures `max_abs_err ≈ 3.81` on this fixture: every element
past the first segment ends up including contributions from segments it
should never have reached across.
