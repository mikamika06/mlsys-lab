#include <cstdio>
#include "sol.hpp"

// FIXED driver: a 128-byte arena, seven allocation requests (one of which
// doesn't fit), then an independent overlap check over whatever offsets
// came back.

constexpr long ARENA_BYTES = 128;
constexpr int N = 7;

int main() {
    long sizes[N]  = {10, 7, 20, 3, 50, 100, 20};
    long aligns[N] = {1, 4, 8, 1, 16, 1, 1};

    bump_reset();

    long offsets[N];
    for (int i = 0; i < N; ++i) {
        offsets[i] = bump_alloc(sizes[i], aligns[i], ARENA_BYTES);
        printf("%ld ", offsets[i]);
    }
    printf("\n");

    int overlaps = 0;
    for (int i = 0; i < N; ++i) {
        if (offsets[i] < 0) continue;
        for (int j = i + 1; j < N; ++j) {
            if (offsets[j] < 0) continue;
            long a0 = offsets[i], a1 = offsets[i] + sizes[i];
            long b0 = offsets[j], b1 = offsets[j] + sizes[j];
            if (a0 < b1 && b0 < a1) ++overlaps;
        }
    }
    printf("overlaps=%d\n", overlaps);
    return 0;
}
