#pragma once

// ============================================================================
// LEARNER implements these two in solve.cpp.
//
// bump_reset(): reset the allocator's internal cursor back to 0 (the start
// of the arena).
//
// bump_alloc(size, align, arena_bytes): bump-allocate `size` bytes, aligned
// to `align` (a power of two), from a fixed `arena_bytes`-byte arena.
//   1. Round the CURRENT cursor up to the next multiple of `align`.
//   2. If that rounded cursor + size > arena_bytes, the allocation does
//      NOT fit: return -1 and leave the cursor UNCHANGED (a failed
//      allocation must not consume any space).
//   3. Otherwise, the allocation begins at the rounded cursor. Advance the
//      cursor to (rounded cursor + size) and return the rounded cursor as
//      the allocation's offset.
//
// Successive successful allocations must never overlap: each returned
// region [offset, offset + size) must be entirely past the end of every
// previously returned region.
// ============================================================================
void bump_reset();
long bump_alloc(long size, long align, long arena_bytes);
