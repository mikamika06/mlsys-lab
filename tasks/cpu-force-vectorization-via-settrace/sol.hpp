#pragma once

// Instrumentation hook, defined in main.cpp: the kernel must call this
// exactly once for every SIMD-width vector instruction it issues -- i.e.
// once per contiguous chunk of `width` elements processed together,
// never once per scalar element and never zero times. main.cpp counts
// the calls in g_vector_ops and prints them alongside the result, so a
// scalar per-element loop shows up as a much larger instruction count
// than a truly vectorized (width-wide) loop.
extern int g_vector_ops;
void op_tick();

// Fused multiply-add, elementwise: out[i] = a[i]*b[i] + c[i] for
// i in [0, n). n is guaranteed to be a multiple of width (a real vector
// register holds exactly `width` float32 lanes -- e.g. width=4 for a
// 128-bit NEON register on Apple Silicon, width=8 for a 256-bit AVX
// register on x86). The implementation MUST process the arrays in
// contiguous chunks of `width` elements and call op_tick() exactly once
// per chunk: that is what "vectorized" means for this task. Calling
// op_tick() once per scalar element means the hardware never actually
// issued a wide vector instruction -- it just executed the same number
// of lanes one at a time.
void fma_vectorized(const float* a, const float* b, const float* c,
                     float* out, int n, int width);
