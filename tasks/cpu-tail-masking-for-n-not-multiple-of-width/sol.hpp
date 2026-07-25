#pragma once

// Modeled SIMD lane width (e.g. 4 lanes of a 128-bit float vector).
constexpr int WIDTH = 4;

// ============================================================================
// Elementwise add c[i] = a[i] + b[i] for i in [0, n), modeled as a
// WIDTH-wide vector loop: process n in chunks of WIDTH lanes at a time
// (the "main loop", i in {0, WIDTH, 2*WIDTH, ...}), THEN handle the
// remaining n % WIDTH elements -- the ones that don't fill a whole
// WIDTH-wide chunk -- with a scalar "tail" loop. n is not guaranteed to
// be a multiple of WIDTH: skip the tail and those trailing elements of
// `c` are simply never written.
// ============================================================================
void vec_add(const float* a, const float* b, float* c, int n);
