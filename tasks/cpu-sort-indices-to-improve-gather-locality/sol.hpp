#pragma once

// Cache access hook, DEFINED in main.cpp: a small deterministic
// set-associative LRU cache. Real hardware cache timing isn't
// reproducible across machines, so this model -- not derived from any
// real C++ pointer, purely from the SYNTHETIC table address below -- is
// the sole source of every miss count the driver prints.
void touch(long byte_addr);

// For cache-modeling purposes, table element k lives at SYNTHETIC byte
// address k*4 (independent of `table`'s real, ordinary C++ pointer,
// which holds the ACTUAL values). touch((long)k * 4) once per table
// element you read.
inline long table_addr(int k) { return (long)k * 4; }

// Gather output[i] = table[indices[i]] for every i in [0, n). You may
// READ the table -- and therefore call touch() -- in ANY order (e.g.
// sorted by index value, to group together reads that land in the same
// cache line), but every index in `indices` must be touched EXACTLY
// ONCE, via touch(table_addr(indices[i])), and output[i] must end up
// holding table[indices[i]] for ITS OWN i regardless of the order you
// read the table in.
void gather_sorted(const float* table, int table_len, const int* indices,
                    int n, float* output);
