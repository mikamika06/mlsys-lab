#pragma once

// A stream+stride hardware prefetcher tracks the stride between
// consecutive memory accesses and, once it sees the SAME stride twice in
// a row, starts issuing prefetches one stride ahead. It keeps doing that
// as long as the stride stays constant and small; a stride that jumps
// around, or one that is at least a full page, defeats it.
//
// Model: pattern k's trace is `addrs[k]`, an array of `lens[k]` byte
// addresses (lens[k] >= 2). The pattern is CAUGHT (1) iff EVERY
// consecutive pair of addresses in the trace has the exact same, nonzero
// stride, AND that stride's absolute value is strictly less than one
// page (4096 bytes). Any trace whose stride varies from step to step
// (random access, pointer chasing) or whose constant stride is >= 4096
// bytes is NOT caught (0).
//
// Write one int (1 or 0) per pattern into out[0 .. num_patterns).
void classify_prefetch(const long* const* addrs, const int* lens, int num_patterns, int* out);
