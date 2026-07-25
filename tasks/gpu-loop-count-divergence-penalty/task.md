## Context

A warp's 32 lanes execute in lockstep: every instruction is issued to
every active lane at once. When a loop's trip count is *data-dependent*
(each lane loops a different number of times, e.g. processing a
variable-length list per lane), the hardware can't let fast lanes move on
early — the whole warp has to keep cycling through the loop body,
predicating off lanes that already finished, until the SLOWEST lane is
done too.

For 32 per-lane trip counts $t_0, \dots, t_{31}$:

- The warp actually runs $\max_i t_i$ iterations — however long the
  slowest lane takes.
- Every lane-iteration where a lane was predicated off (already
  finished, but the warp kept cycling anyway) is wasted work: out of
  $32 \times \max_i t_i$ total lane-iteration slots, only $\sum_i t_i$
  did real work. The rest —

$$\text{wasted} = 32 \cdot \max_i t_i \;-\; \sum_i t_i$$

— is silicon spent on nothing.

## Task

Write a CUDA-C kernel (single thread — this derives two numbers from a
fixed array, it doesn't need 32 real lanes to do it):

```cpp
__global__ void divergence_penalty(float* out, const float* trips, int warp_size);
```

`trips[0..warp_size)` holds one warp's 32 per-lane trip counts. Compute
`out[0] = max(trips)` (serialized iterations) and
`out[1] = warp_size * max(trips) - sum(trips)` (wasted lane-iterations).

## Example

With `trips[i] = (i * 7) % 11` for `i` in `0..31` — trip counts cycling
through every value `0..10` roughly 3 times, spread across the warp:

```
max(trips) = 10
sum(trips) = 161
out[0] = 10
out[1] = 32*10 - 161 = 159
```

The warp needed only $161$ real lane-iterations of work but had to run
$320$ lane-iteration slots ($32$ lanes $\times$ $10$ iterations each) to
let its slowest lane (trip count $10$) finish — $159$ of those slots did
nothing.

## What the gate checks

The grader parses your `.cu` with the CUDA-C frontend and runs it (single
thread) on the software GPU over the fixed 32-value fixture, requiring
`max_abs_err <= 1e-6` against both numbers computed directly in numpy.
Using the AVERAGE trip count instead of the max (a plausible but wrong
mental model — "the warp takes about as long as a typical lane") gets
both numbers wrong whenever the trip counts aren't all equal, which they
never are in this fixture. The empty starter leaves both outputs at their
`-1.0` sentinel.
