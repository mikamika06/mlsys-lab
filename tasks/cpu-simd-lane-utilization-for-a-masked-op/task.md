## Context

SIMD hardware (NEON, AVX) executes a fixed-width vector instruction for
a whole group of lanes at once, active or not — a masked/predicated op
with only 1 active lane out of 8 still costs a full 8-lane instruction.
The one thing software CAN do cheaply is check a group's mask ahead of
time and skip the instruction entirely when every lane in it is
inactive. Lane utilization measures how much of the work actually paid
for was useful.

## Task

Implement

```cpp
double simd_lane_utilization(const bool* mask, int n, int width);
```

`mask[0..n)` is grouped into `n/width` groups of `width` consecutive
lanes. Compute `total_active` (count of `true` entries) and
`groups_executed` (count of groups with **at least one** `true` lane —
an all-`false` group is skipped and does not count). Return
`total_active / (groups_executed * width)`.

## Example

With `width = 8` and 4 groups: group 0 is fully active (`8/8`), group 1
has 2 active lanes, group 2 is fully inactive (skipped — contributes 0 to
both the numerator and the group count), group 3 has 5 active lanes.
`total_active = 8+2+0+5 = 15`, `groups_executed = 3` (group 2 doesn't
count), so `utilization = 15 / (3*8) = 0.625`.

## What the gate checks

`max_abs_err` on the printed utilization for one fixed 32-lane mask.
Counting the fully-inactive group as executed (dividing by `4*8`
instead of `3*8`), or using `n` instead of `groups_executed * width` as
the denominator, gives `0.46875` or `0.469` instead of `0.625`; a
starter returning `0.0` fails outright.
