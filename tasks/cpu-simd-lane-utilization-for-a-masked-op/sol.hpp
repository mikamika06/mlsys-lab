#pragma once

// A predicated (masked) SIMD op processes elements in fixed-width
// groups of `width` lanes (n is a multiple of width). Hardware executes
// a group as a single vector instruction if ANY lane in it is active;
// software can check the mask first and skip a group whose lanes are
// ALL inactive entirely (it costs nothing and does no useful work). A
// group that runs at all still costs a full `width`-lane instruction,
// even if only 1 of its lanes is actually active -- the other lanes are
// wasted.
//
// Given mask[0..n) (true = lane is active) grouped into n/width groups
// of `width` consecutive lanes each, compute:
//   total_active     = number of true entries in mask (useful lane-ops)
//   groups_executed  = number of groups containing AT LEAST one true
//                       lane (a fully-false group is skipped and does
//                       not count)
//   utilization      = total_active / (groups_executed * width)
//                       (fraction of the executed groups' total lane
//                       capacity that was actually useful)
// Return utilization.
double simd_lane_utilization(const bool* mask, int n, int width);
