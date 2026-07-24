#pragma once

// ---------------------------------------------------------------------------
// PROVIDED (defined in main.cpp): a deterministic cache-line-ownership
// model, standing in for real MESI cross-core coherence traffic (real
// hardware cache-invalidation counts are not reproducible, so this task
// never times anything or reads hardware counters).
//
// Every 64-byte-aligned cache LINE has a single "owning" thread: whichever
// thread wrote to it most recently. Call report_write(thread_id, byte_addr)
// once per simulated write. If the line at byte_addr currently has no
// owner, thread_id simply becomes the owner (no invalidation -- this is
// the line's very first write). If the line's current owner is a
// DIFFERENT thread, that is a real cross-core cache-line invalidation:
// the global counter is bumped and thread_id becomes the new owner. If the
// line's current owner already IS thread_id, nothing happens (cheap,
// same-core reuse).
// ---------------------------------------------------------------------------
constexpr long CACHE_LINE_BYTES = 64;

void report_write(int thread_id, long byte_addr);
long total_invalidations();
void reset_invalidations();

// ---------------------------------------------------------------------------
// LEARNER IMPLEMENTS.
//
// Simulate `num_threads` threads, each incrementing its own private
// counter `num_increments` times, ROUND-ROBIN interleaved (thread 0 does
// one increment, then thread 1, ..., then num_threads-1, then back to
// thread 0, ...) -- the classic worst case for false sharing, and exactly
// how independent cores really do race for a shared line under
// contention.
//
// simulate_aos_invalidations: each thread's 4-byte int counter is packed
//   TIGHTLY at byte address `thread_id * 4` (as in `int counters[N];`), so
//   several consecutive threads' counters land on the SAME 64-byte cache
//   line.
// simulate_padded_invalidations: each thread's counter instead starts at
//   byte address `thread_id * CACHE_LINE_BYTES` -- one whole cache line per
//   thread, so no two threads' counters ever share a line.
//
// For each function: reset_invalidations(), then call
// report_write(thread_id, byte_address) once per simulated increment, in
// round-robin thread order, for num_threads * num_increments total calls
// using the byte-address formula for that layout. Return
// total_invalidations() at the end.
// ---------------------------------------------------------------------------
long simulate_aos_invalidations(int num_threads, int num_increments);
long simulate_padded_invalidations(int num_threads, int num_increments);
