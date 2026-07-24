## Context

A warp is 32 lanes executing in lockstep. `__shfl_up_sync(mask, v, delta)`
lets lane `l` read the value `v` held by lane `l - delta` in the *same*
instruction - no shared memory, no barrier, no round trip through SMEM.

That makes the Hillis-Steele scan almost free inside a warp:

```
for delta in 1, 2, 4, 8, 16:
    up = __shfl_up_sync(0xffffffff, v, delta)
    if (lane >= delta) v += up;
```

After the five steps every lane holds the inclusive prefix sum of its warp.

## Task

Implement `inclusive_scan_warp(t, n)`: an in-place inclusive prefix sum over
global memory, computed independently inside each warp.

The simulator models a shuffle exactly as hardware does - warp-synchronous -
so the kernel is a **generator**: yield the request and receive the value.

```python
up = yield t.shfl_up(v, delta)
```

`t.lane` is the lane index within the warp. A lane whose source is below lane
0 keeps its own value, matching `__shfl_up_sync`.

## Example

With `n = 64` (two warps), element 0..31 become the running sum of the first
warp and 32..63 the running sum of the second - the sums never cross the warp
boundary.

## What the gate checks

- `max_abs_err <= 1e-12` against a per-warp `np.cumsum` oracle.
- `smem_waves <= 0`: solving it through shared memory instead of shuffles fails.
