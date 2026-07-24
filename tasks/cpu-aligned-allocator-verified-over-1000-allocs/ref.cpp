#include "sol.hpp"
#include <cstdint>

void bump_init(BumpState& s, unsigned char* arena, std::size_t arena_size) {
    s.arena = arena;
    s.arena_size = arena_size;
    s.offset = 0;
}

void* aligned_alloc_bump(BumpState& s, std::size_t size, std::size_t align) {
    std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(s.arena) + s.offset;
    std::uintptr_t aligned_addr = (addr + (align - 1)) & ~(align - 1);
    std::size_t new_offset = static_cast<std::size_t>(aligned_addr - reinterpret_cast<std::uintptr_t>(s.arena)) + size;
    if (new_offset > s.arena_size) {
        return nullptr;
    }
    s.offset = new_offset;
    return reinterpret_cast<void*>(aligned_addr);
}
