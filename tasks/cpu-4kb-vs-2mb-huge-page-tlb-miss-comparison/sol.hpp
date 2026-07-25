#pragma once
// TLB access hook, DEFINED in main.cpp: touch() records one memory access
// against a small deterministic set-associative LRU TLB, currently
// configured with S=64 sets and W=4 ways (256 entries total).
// reset_tlb(page_size) reconfigures the TLB for a fresh page size in
// bytes and clears all state; miss_count() reads the number of misses
// recorded since the last reset_tlb() call. Real hardware TLB behaviour
// is not reproducible across machines, so this model is the sole source
// of every miss count below.
void touch(long byte_addr);
void reset_tlb(long page_size);
long miss_count();

// Touch a `stride`-byte-spaced working set of `count` elements starting
// at byte address `base` (i.e. addresses base + i*stride for i in
// [0, count)), REPEATED `passes` times in the same ascending order each
// pass -- once against a TLB reset with 4 KiB (4096-byte) pages, once
// against a TLB reset with 2 MiB (2097152-byte) "huge" pages. The exact
// same access pattern is replayed both times; only the page size (and
// therefore how many distinct TLB entries the working set needs)
// differs. Write the two resulting miss counts into out[0] (4 KiB) and
// out[1] (2 MiB).
void tlb_miss_pair(long base, long stride, int count, int passes, long* out);
