#pragma once
// 4x4 float32 transpose using real ARM NEON vzipq_f32 shuffles (<arm_neon.h>).
//
// `in` and `out` are row-major 4x4 tiles (16 floats each; row r occupies
// in[4*r .. 4*r+4)). Must write out[c*4 + r] = in[r*4 + c] for every
// r, c in [0,4) — the exact transpose — by composing vzipq_f32 shuffles on
// the four loaded rows, not by scalar index swapping.
//
// vzipq_f32(a, b) interleaves two 4-lane vectors and returns a pair:
//   .val[0] = [a0, b0, a1, b1]     (interleave of the low halves)
//   .val[1] = [a2, b2, a3, b3]     (interleave of the high halves)
//
// The reference loads rows r0..r3, zips (r0, r2) and (r1, r3) to get two
// half-transposed pairs, then zips those pairs together lane-by-lane to
// land the four fully-transposed output rows.
void transpose4x4(const float* in, float* out);
