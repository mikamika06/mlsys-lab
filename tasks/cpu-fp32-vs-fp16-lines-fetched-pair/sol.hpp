#pragma once

// A sequential scan of n elements of `width` bytes each, starting at
// byte address `base`, touches element i at address base + i*width. Two
// addresses fall in the same cache line iff addr / line_bytes is equal.
//
// compare_fp32_fp16_lines: run that scan twice over the SAME n and base
// -- once with width=4 (fp32 elements) and once with width=2 (fp16
// elements) -- and write the number of distinct cache lines touched by
// each into out[0] (fp32) and out[1] (fp16).
void compare_fp32_fp16_lines(long base, int n, int line_bytes, long* out);
