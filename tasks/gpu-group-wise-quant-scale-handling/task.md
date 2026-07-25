## Context

A single scale factor for an entire quantized tensor wastes precision:
one outlier row forces every other row's codes to spread across a wider
range than they need. **Group-wise quantization** instead splits the
tensor into fixed-size groups of `G` consecutive elements, giving each
group its *own* scale — a quiet group with small values gets a small
scale (and keeps its precision), a group with a big outlier gets a big
scale, independently.

Dequantizing has to look up the *right* group's scale for every element:
element `i` belongs to group `i / G`, and nowhere else. Using the wrong
group index — off by one group, or reusing a single shared scale for the
whole tensor — silently corrupts every element in the misattributed
groups, and it's especially brutal for a "quiet" group whose true scale
is tiny: multiply it by a neighboring group's much larger scale and its
values are crushed into noise.

## Task

Implement, in real CUDA-C:

```cuda
__global__ void dequant_groupwise(float* out, const float* codes, const float* scale, int G, int n);
```

For `i = blockIdx.x*blockDim.x + threadIdx.x`, guarded by `i < n`:
`out[i] = codes[i] * scale[i / G]`.

## Example

`G=16`, group scales `[0.1, 1.0, 5.0, 0.01]`: element `20` belongs to
group `20/16 = 1`, so it's scaled by `1.0`; element `50` belongs to group
`50/16 = 3` (the "quiet" group, scale `0.01`) — using group `2`'s scale
(`5.0`) instead would overshoot that element by a factor of 500.

## What the gate checks

`max_abs_err <= 1e-9` on 64 fixed codes (`-7..7`, cycling) split into 4
groups of 16 with scales `[0.1, 1.0, 5.0, 0.01]` — deliberately spread
across two orders of magnitude so any group-index mistake produces a
large, obvious error rather than a subtle one. Using a single global
scale, computing the group index as `i % G` instead of `i / G`, or an
off-by-one on the group boundary, all misattribute at least one group's
elements and fail the match.
