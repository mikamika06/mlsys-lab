#include "sol.hpp"

void bump_init(BumpState& s, unsigned char* arena, std::size_t arena_size) {
    s.arena = arena;
    s.arena_size = arena_size;
    s.offset = 0;
}

void* aligned_alloc_bump(BumpState& s, std::size_t size, std::size_t align) {
    // your code here
    return nullptr;
}
