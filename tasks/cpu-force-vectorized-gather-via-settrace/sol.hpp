#pragma once

// Cache access hook, DEFINED in main.cpp: a small deterministic
// set-associative LRU cache that also counts every call. Real hardware
// cache/load counts aren't reproducible across machines, so this model
// -- fed purely by the SYNTHETIC addresses below -- is the sole source
// of every touch/miss count the driver prints.
void touch(long byte_addr);

// For cache-modeling purposes, table element k lives at SYNTHETIC byte
// address k*4 (independent of `table`'s real, ordinary C++ pointer,
// which holds the ACTUAL values). touch(table_addr(k)) once per element
// value k you actually fetch from memory.
inline long table_addr(int k) { return (long)k * 4; }

// Gather output[i] = table[indices[i]] for every i in [0, n).
//
// A real vectorized/SIMD gather that sees the SAME index value again
// within its working set services the repeat from a register/lane it
// already holds, instead of issuing a fresh memory access. Model that
// here with your OWN value cache (e.g. two arrays of size table_len:
// "have we ever fetched index k" and "what value did we fetch for k"):
// call touch(table_addr(indices[i])) ONLY the FIRST time a given index
// VALUE is looked up anywhere during the call; every later repeat of
// that same value must be served from your own cache, with NO
// additional call to touch(). output[i] must still equal
// table[indices[i]] for every i, repeats included.
void gather_dedup(const float* table, int table_len, const int* indices,
                   int n, float* output);
