#pragma once

// ============================================================================
// LEARNER defines a struct in solve.cpp:
//
//   struct ThreadData {
//       long counter;      // MUST stay the first member
//       /* padding fields you add, appended AFTER counter */
//   };
//
// and implements thread_data_sizeof() to return sizeof(ThreadData) — the
// real compiler's own answer, not a modeled one.
//
// The driver treats an array `ThreadData data[4]` as living at address 0
// (one array element per thread) and derives each thread's counter address
// as `thread_id * thread_data_sizeof()`. Pad ThreadData so that no two of
// the 4 threads' counters land in the same 64-byte cache line — i.e. make
// sizeof(ThreadData) a multiple of 64.
// ============================================================================
int thread_data_sizeof();
