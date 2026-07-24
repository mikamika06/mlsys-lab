#include "sol.hpp"
#include <cstdint>
#include <cstdio>

static unsigned char g_arena[1 << 20];

int main() {
    BumpState s;
    bump_init(s, g_arena, sizeof(g_arena));

    int nonnull_count = 0;
    int aligned_count = 0;
    int nonoverlap_count = 0;
    std::uintptr_t prev_end = 0;
    bool have_prev = false;

    for (int i = 0; i < 1000; i++) {
        std::size_t align = static_cast<std::size_t>(1) << (i % 7); // 1,2,4,8,16,32,64
        std::size_t size = static_cast<std::size_t>(i % 37) + 1;    // 1..37
        void* p = aligned_alloc_bump(s, size, align);
        if (p != nullptr) {
            nonnull_count++;
            std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(p);
            if (addr % align == 0) aligned_count++;
            if (!have_prev || addr >= prev_end) nonoverlap_count++;
            prev_end = addr + size;
            have_prev = true;
        }
    }

    printf("%d %d %d\n", nonnull_count, aligned_count, nonoverlap_count);
    printf("%zu\n", s.offset);
    return 0;
}
