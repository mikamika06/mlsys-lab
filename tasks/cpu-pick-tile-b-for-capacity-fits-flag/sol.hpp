#pragma once

// ============================================================================
// Deterministic fully-associative LRU cache model (FIXED — defined in
// main.cpp): `capacity_bytes / 64` lines of 64 bytes each, true LRU
// eviction. touch_byte(addr) simulates a read/write of one byte at that
// address, counting a MISS whenever that line was not already resident.
// ============================================================================
void reset_cache();
void touch_byte(long addr);
long miss_count();

// ============================================================================
// A blocked kernel (think the inner loop of a blocked matmul) keeps THREE
// B x B float tiles resident in cache at once. Derive the LARGEST integer
// tile edge B such that all three tiles together fit within
// `capacity_bytes` of cache:
//
//   3 * B * B * elem_size <= capacity_bytes
//
// (elem_size is the byte size of one tile element, e.g. sizeof(float).)
// ============================================================================
int derive_tile_b(long capacity_bytes, int elem_size);
