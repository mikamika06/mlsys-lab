#include <cstdio>
#include "sol.hpp"

// ============================================================================
// FIXED set-mapping oracle. Ground truth: which set does byte address
// `addr` fall into for a cache with `line_bytes`-byte lines and `sets`
// sets? This is independent of any candidate's derived stride.
// ============================================================================
int set_of(long addr, int line_bytes, int sets) {
    long line = addr / line_bytes;
    return static_cast<int>(line % sets);
}

namespace {
struct Case {
    int line_bytes;
    int sets;
    int num_accesses;
};

// Deterministic, fixed test cases (mixed power-of-two and non-power-of-two
// set counts, so a formula that only happens to work for powers of two
// still gets caught).
constexpr Case CASES[] = {
    {64, 8, 6},
    {32, 16, 6},
    {64, 4, 6},
    {16, 32, 6},
    {128, 8, 6},
    {64, 6, 6},
    {256, 3, 6},
};
constexpr int NUM_CASES = sizeof(CASES) / sizeof(CASES[0]);
} // namespace

int main() {
    int total_agree = 0;
    for (int c = 0; c < NUM_CASES; ++c) {
        int line_bytes = CASES[c].line_bytes;
        int sets = CASES[c].sets;
        int num_accesses = CASES[c].num_accesses;

        long stride = collision_stride(line_bytes, sets);
        int base_set = set_of(0, line_bytes, sets);

        int agree = 1;
        for (int k = 0; k < num_accesses; ++k) {
            long addr = static_cast<long>(k) * stride;
            if (set_of(addr, line_bytes, sets) != base_set) {
                agree = 0;
            }
        }

        printf("case=%d line_bytes=%d sets=%d stride=%ld agree=%d\n",
               c, line_bytes, sets, stride, agree);
        total_agree += agree;
    }
    printf("total_agree=%d\n", total_agree);
    return 0;
}
