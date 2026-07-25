#pragma once

// Cache access hook, DEFINED in main.cpp: a small deterministic
// set-associative LRU cache. touch() records one byte-address access;
// reset_cache() clears it and (re)configures it to a specific
// (line_bytes, sets, ways); miss_count() reads the miss count since the
// last reset_cache(). Real hardware cache timing isn't reproducible
// across machines, so this model is the sole source of every miss count.
void touch(long byte_addr);
void reset_cache(int line_bytes, int sets, int ways);
long miss_count();

// A multi-head projection weight stack: H heads, each owning `row_bytes`
// contiguous bytes (one 4-byte float per row of an R-row stack) PLUS
// `pad` bytes of padding appended after it, so head h's row r (a single
// float) lives at byte address
//   h * (row_bytes + pad) + r * 4
//
// When `row_bytes` is a power of two (the realistic case: it IS d_model
// worth of rows, itself a power of two), pad = 0 makes every head's
// stride an exact multiple of the cache's (line_bytes * sets) span --
// every head lands in the SAME handful of cache sets, so gathering all H
// heads' row-r value TWICE in a row (once to compute something, once
// again right after to use it) evicts and re-fetches lines that never
// needed to leave the cache, even though the whole H-element working set
// is tiny.
//
// Search pad over {0, 4, 8, ..., max_pad_bytes} (inclusive, step 4
// bytes). For each candidate, measure how many misses this exact
// two-pass, single-row access pattern produces against a FRESH cache:
//   reset_cache(line_bytes, sets, ways);
//   for h in [0, H): touch(h * (row_bytes + pad) + 0);   // pass 1
//   for h in [0, H): touch(h * (row_bytes + pad) + 0);   // pass 2, same addresses
//   candidate_misses = miss_count();
// Return the pad achieving the FEWEST candidate_misses; break ties by
// returning the SMALLEST such pad. Use touch()/reset_cache()/miss_count()
// for every trial -- never hand-model the cache yourself.
int choose_padding_bytes(int H, int row_bytes, int line_bytes, int sets,
                          int ways, int max_pad_bytes);
