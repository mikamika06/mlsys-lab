#pragma once

// Pinned cache-line size in bytes (defined in main.cpp).
extern const int LINE_BYTES;

// LEARNER IMPLEMENTS.
//
// Total bytes of memory-bus traffic to fully overwrite a buffer of
// `total_bytes` bytes using ORDINARY ("temporal") stores, starting with
// none of the buffer's lines cache-resident.
//
// Every line the store touches for the first time isn't resident, so
// before the CPU can write even one byte of it, the cache controller
// must fetch the WHOLE line from memory to gain exclusive ownership
// (a Read-For-Ownership, RFO) -- even though every byte in that line is
// about to be overwritten anyway. Later, when that now-dirty line is
// evicted, the whole line is written BACK to memory too. So each line
// costs one full-line READ (the RFO) plus one full-line WRITE (the
// eventual writeback): round total_bytes up to a whole number of
// LINE_BYTES-sized lines, then charge 2 line-widths per line.
long temporal_store_traffic(long total_bytes);

// Total bytes of memory-bus traffic to fully overwrite the same buffer
// using NON-TEMPORAL ("streaming") stores: these bypass the cache
// entirely (no RFO, and nothing to write back later, since the data
// never enters the cache) and go straight to memory through the
// write-combining buffers, so each line costs only one full-line WRITE:
// round total_bytes up to a whole number of LINE_BYTES-sized lines,
// then charge 1 line-width per line.
long nontemporal_store_traffic(long total_bytes);
