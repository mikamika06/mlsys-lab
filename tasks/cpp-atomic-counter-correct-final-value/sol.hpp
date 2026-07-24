#pragma once

// Spawn `num_threads` real std::thread workers. Each worker increments a
// SINGLE SHARED counter exactly `increments_per_thread` times. Join every
// thread, then return the counter's final value.
//
// The increments must be atomic (e.g. std::atomic<long>::fetch_add) so that
// no update from any thread is ever lost to a race, no matter how the OS
// schedules/interleaves the threads. That guarantee is what makes the
// result deterministic and always exactly
//     num_threads * increments_per_thread
long atomic_counter_final_value(int num_threads, int increments_per_thread);
