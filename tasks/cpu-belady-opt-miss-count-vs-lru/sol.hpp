#pragma once
// Simulate Belady's OPT (MIN) cache replacement policy over a reference
// string refs[0..n) with a fully-associative cache of `capacity` lines,
// and return the number of MISSES.
//
// OPT is the clairvoyant policy: on a miss with a full cache, evict
// whichever resident line is used FURTHEST in the future (or never used
// again -- that beats any finite next-use distance). It is provably the
// minimum possible miss count for that reference string and capacity, so
// it's the standard yardstick every other policy (LRU, FIFO, ...) is
// measured against.
int belady_opt_misses(const int* refs, int n, int capacity);
