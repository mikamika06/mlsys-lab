#pragma once

// Given a trace of `n` byte addresses `addrs[0..n)`, an access at index i
// is a REUSE if the exact same address value already appeared at some
// strictly earlier index j < i in the trace -- i.e. this is not the first
// time the trace has ever touched that address. This is the simplest
// possible notion of temporal locality: it says nothing about cache
// capacity, associativity, or eviction -- only "have we seen this address
// before, at any distance". The first occurrence of any address is never
// a reuse.
//
// Return the total number of reuse accesses in the trace.
long long count_reuses(const long* addrs, int n);
