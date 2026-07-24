#pragma once

// ============================================================================
// LEARNER implements this in solve.cpp.
//
// modeled_vector_speedup: model the instruction-count speedup of a
// width-`width` SIMD dot product over an `n`-element scalar dot product
// (e.g. `width=4` for NEON float32x4, `width=8` for AVX2 256-bit float,
// `width=16` for AVX-512).
//
// The scalar baseline issues exactly `n` fused-multiply-add instructions
// (one element per instruction). The vectorized version issues
// `n / width` (integer division) full-width vector FMA instructions,
// covering `(n / width) * width` elements, and then falls back to `n %
// width` ordinary scalar instructions for the leftover tail that doesn't
// fill a whole vector register. Total vector-path instruction count:
//     vector_instrs = n / width + n % width
// Return the modeled speedup as an instruction-count ratio:
//     modeled_vector_speedup = (double)n / (double)vector_instrs
// When `n` is an exact multiple of `width` this is exactly `width` (the
// ideal case, zero tail overhead); any nonzero remainder pulls it below
// `width`, and when `n < width` (everything is tail) it collapses to 1.0 --
// no vectorization benefit at all.
// ============================================================================
double modeled_vector_speedup(int n, int width);
