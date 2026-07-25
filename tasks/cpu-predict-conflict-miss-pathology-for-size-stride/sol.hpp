#pragma once

// Pinned L1 shape (fixed, matches a typical real 32KB 8-way L1D):
constexpr int LINE_BYTES = 64;
constexpr int NUM_SETS = 64;
constexpr int WAYS = 8;
// capacity = LINE_BYTES * NUM_SETS * WAYS = 32768 bytes

// ============================================================================
// A loop sweeps `array_size` bytes of an array with a fixed `stride` (in
// bytes, always a multiple of LINE_BYTES here): touching addresses
// 0, stride, 2*stride, ..., up to array_size, once per sweep, repeated
// every time the loop containing it runs again.
//
// Predict whether this (array_size, stride) access pattern is
// PATHOLOGICAL for the pinned L1 above: even though the total bytes
// touched may fit comfortably within the cache's total capacity, a
// stride that is a multiple of (LINE_BYTES * NUM_SETS) / k for some
// small k funnels every touched line into only a handful of the 64
// sets, so that set alone sees more DISTINCT lines than it has WAYS to
// hold -- repeat the sweep, and every one of those lines gets evicted
// and re-fetched every time, no matter how much spare capacity the
// OTHER 63 sets have sitting idle.
//
// Formally: let d = (stride / LINE_BYTES) mod NUM_SETS. Stepping a set
// index by d (mod NUM_SETS) visits exactly NUM_SETS / gcd(d, NUM_SETS)
// distinct sets, cycling repeatedly (treat d == 0 as visiting exactly 1
// set). Let n = array_size / stride be the number of elements one sweep
// touches. Return 1 (pathological) if
//   n / (NUM_SETS / gcd(d, NUM_SETS))  >  WAYS
// i.e. more distinct lines land in the busiest set than it has ways for.
// Return 0 (benign) otherwise.
// ============================================================================
int classify_pathological(long array_size, long stride);
