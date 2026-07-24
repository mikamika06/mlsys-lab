#pragma once

// The driver models N threads, each repeatedly writing its OWN 8-byte
// counter, laid out as `counter[thread_id]` in an array with a fixed
// byte stride between consecutive threads' counters:
//
//   stride = 8 + counter_pad_bytes()
//   thread tid's counter address = tid * stride
//
// Return the number of PADDING bytes to insert after each thread's
// 8-byte counter so that no two of the driver's threads' counters ever
// land in the same 64-byte cache line -- i.e. so `stride` ends up a
// multiple of 64.
int counter_pad_bytes();
