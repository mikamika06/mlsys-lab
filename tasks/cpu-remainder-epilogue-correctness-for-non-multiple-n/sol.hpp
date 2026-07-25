#pragma once

// Compute out[i] = s * in[i] for every i in [0, n), using a 4-way
// unrolled main loop (processing 4 elements per iteration, for the
// largest multiple of 4 that is <= n) followed by a scalar EPILOGUE loop
// that handles the remaining n % 4 elements individually. Every element
// of out[0..n) must be written -- an unrolled loop that stops after the
// last full group of 4 and skips the remainder leaves those tail
// elements wrong.
void scale_unrolled(const float* in, int n, float s, float* out);
