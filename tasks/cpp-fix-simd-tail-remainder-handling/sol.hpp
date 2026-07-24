#pragma once

// Sum n int32 elements. May use NEON (<arm_neon.h>) to process 4 at a
// time, but MUST handle every length correctly, including one that is not
// a multiple of 4 (the "tail" / remainder elements).
long long simd_sum(const int* data, int n);
