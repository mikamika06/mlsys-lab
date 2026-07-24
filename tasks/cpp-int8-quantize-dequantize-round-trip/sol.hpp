#pragma once

// Quantize each of the n floats in `data` to a signed int8 using
// round-half-to-even (banker's rounding, IEEE-754's default), clamped to
// [-128, 127], then immediately dequantize each int8 back to a float.
// Write the n reconstructed floats into `out` (caller-allocated, room for
// n floats).
//
//   q  = clamp(round_half_to_even(x / scale) + zero_point, -128, 127)
//   x_hat = (q - zero_point) * scale
void quantize_dequantize(const float* data, int n, float scale, int zero_point, float* out);
