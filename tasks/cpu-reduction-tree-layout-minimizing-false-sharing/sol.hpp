#pragma once

// The driver models a 16-thread binary-tree reduction: thread tid holds
// an 8-byte partial-sum slot at byte address `tid * stride`, where
// `stride = 8 + slot_pad_bytes()`. The reduction runs in 4 rounds; in
// round r (r = 0..3), every thread whose id is a multiple of 2^(r+1)
// writes its own slot (combining in its neighbour's partial sum at
// distance 2^r). So round 0 has 8 writers (every even tid), round 1 has
// 4 (multiples of 4), round 2 has 2 (multiples of 8), round 3 has 1
// (tid 0) -- 15 writes total, in round order.
//
// Return the SMALLEST number of padding bytes such that, across that
// entire 15-write schedule, no two DIFFERENT threads' slots ever land
// in the same 64-byte cache line (i.e. no write's line is currently
// "owned", from an earlier write, by some other thread).
int slot_pad_bytes();
