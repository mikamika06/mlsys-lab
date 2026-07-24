#pragma once
// Decode a fixed-width integer from a raw bit pattern.
//
// Interpret the low `width` bits of `bits` as an integer of that width.
//  - If is_signed != 0, decode as TWO'S COMPLEMENT: the most-significant of the
//    `width` bits carries weight -2^(width-1), so patterns with that bit set are
//    negative.
//  - If is_signed == 0, decode as a plain unsigned magnitude.
// Any bits at or above position `width` in `bits` must be ignored.
// `width` is in the range 1..32.
long long twos_complement_value(unsigned long long bits, int width, int is_signed);
