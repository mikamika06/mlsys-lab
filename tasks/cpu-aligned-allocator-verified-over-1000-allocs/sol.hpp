#pragma once
#include <cstddef>

// A simple bump allocator over a fixed byte arena. `state.arena` points to
// `state.arena_size` bytes of raw storage; `state.offset` is the
// allocator's current bump position, measured from the start of the
// arena.
struct BumpState {
    unsigned char* arena;
    std::size_t arena_size;
    std::size_t offset;
};

void bump_init(BumpState& s, unsigned char* arena, std::size_t arena_size);

// Round the CURRENT ADDRESS (arena + offset), not just the offset, up to
// the next multiple of `align` (align is always a power of two), reserve
// `size` bytes starting there, advance state.offset past the allocation,
// and return the aligned pointer. Return nullptr (without modifying
// state.offset) if the allocation would not fit within the arena.
void* aligned_alloc_bump(BumpState& s, std::size_t size, std::size_t align);
