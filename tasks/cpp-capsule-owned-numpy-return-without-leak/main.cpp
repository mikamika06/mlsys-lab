#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>
#include "sol.hpp"

// Instrumented allocator: counts every allocation and every free so a leak
// or a double-free shows up as a wrong number (or a crash) instead of
// silently passing.
static int g_alloc_count = 0;
static int g_free_count = 0;

unsigned char* arena_alloc(int n) {
    g_alloc_count++;
    return (unsigned char*)std::malloc((size_t)n);
}

void arena_free(unsigned char* p) {
    if (p != nullptr) g_free_count++;
    std::free(p);
}

static void print_cap(const Capsule& c) {
    printf("size=%d data=", c.size);
    if (c.data == nullptr) {
        printf("null");
    } else {
        for (int i = 0; i < c.size; i++) printf("%d,", (int)c.data[i]);
    }
    printf("\n");
}

// FIXED driver. Exercises every ownership-transfer path a "return a
// capsule-owned buffer" API must survive: reallocation inside a growing
// vector (forces move-construction of existing elements), an explicit
// move-assign over a live capsule (must free the old buffer first), a
// temporary destroyed at end of its own scope, and a move-construct that
// must leave the source empty. Every capsule is destroyed (the whole
// scenario runs inside one block) before alloc/free counts are printed, so
// a leak or a double-free is visible in the final numbers.
int main() {
    {
        std::vector<Capsule> caps;
        for (int i = 0; i < 5; i++) {
            caps.push_back(make_capsule(4 + i, i + 1));
        }

        caps[0] = std::move(caps[4]);  // must free caps[0]'s old buffer first
        {
            Capsule tmp = make_capsule(3, 9);
            (void)tmp;
        }  // tmp destroyed here

        Capsule moved_out = std::move(caps[2]);  // caps[2] must become empty

        print_cap(caps[0]);
        print_cap(caps[1]);
        print_cap(caps[2]);
        print_cap(caps[3]);
        print_cap(moved_out);
    }  // every capsule above is destroyed by here

    printf("alloc=%d free=%d\n", g_alloc_count, g_free_count);
    return 0;
}
