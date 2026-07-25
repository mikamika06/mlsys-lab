#pragma once

// Deterministic set-associative LRU cache probe (FIXED, DEFINED in
// main.cpp): 64-byte lines, 32 sets, 4-way (8192 bytes total). Real
// hardware cache timing is not reproducible across machines, so this
// model -- not the CPU's actual cache -- is the sole source of every
// miss count the driver prints. reset_cache() clears all state; touch()
// records one byte-address access; misses() returns the miss count
// accumulated since the last reset.
void reset_cache();
void touch(long byte_addr);
long misses();

// ---------------------------------------------------------------------
// Contiguous vectorised load: read `n` elements of `elem_bytes` bytes
// each, starting at byte address `base`, strictly in order, `vec_width`
// elements at a time. A real SIMD load instruction issues exactly ONE
// load uop per full vector register regardless of how many bytes it
// spans -- call touch() exactly ONCE per chunk, at the chunk's first
// byte address (base + chunk_index * vec_width * elem_bytes). `n` is
// always an exact multiple of `vec_width`. Return the number of load
// uops issued: n / vec_width.
long contiguous_load(long base, int n, int vec_width, int elem_bytes);

// ---------------------------------------------------------------------
// Scalar gather load: read `n` elements of `elem_bytes` bytes each, base
// address `base`, visited in the order given by the permutation `idx`
// (idx[k] is the k-th element index to fetch, 0 <= idx[k] < n). A
// gather has no vector hardware support for a scattered address
// pattern, so the CPU computes each element's address and issues its
// own load uop -- call touch() exactly once per element, in order, at
// byte address base + (long)idx[k] * elem_bytes. Return the uop count: n.
long gather_load(long base, const int* idx, int n, int elem_bytes);
