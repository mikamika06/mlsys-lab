#pragma once
#include <cstddef>

constexpr int NUM_THREADS = 8;
constexpr int NUM_BINS = 4; // per-thread local token-count bins (int64 each)

// ============================================================================
// Deterministic false-sharing / coherence model (FIXED — do not modify;
// defined in main.cpp). Tracks, per 64-byte cache line, which thread last
// WROTE to it. write_counter(thread_id, addr) performs a write to that
// byte address on behalf of `thread_id`; if the line containing `addr`
// has been written before AND its last writer was a DIFFERENT thread,
// that's an INVALIDATION -- a real MESI-coherent cache would have to
// re-acquire exclusive ownership of the line, invalidating the other
// core's cached copy, exactly as if it were a genuine cache miss.
// ============================================================================
void reset_coherence();
void write_counter(int thread_id, long addr);
long invalidation_count();

// ============================================================================
// NUM_THREADS threads each keep their OWN local histogram of NUM_BINS
// int64 token-count bins — a classic sharded-accumulation pattern: each
// thread tallies token categories for the chunk of text it is
// responsible for, merging all threads' histograms only once, at the
// very end. Derive the byte STRIDE between the start of thread t's block
// of NUM_BINS counters and thread (t+1)'s block, so that no two threads'
// blocks ever land on the same 64-byte cache line (the driver places
// thread 0's block at a 64-byte-aligned base address, thread t's block
// at base + t * thread_block_stride()).
// ============================================================================
size_t thread_block_stride();
