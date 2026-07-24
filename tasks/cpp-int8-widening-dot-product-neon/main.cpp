#include <cstdio>
#include <cstdint>
#include <vector>
#include "sol.hpp"

// PROVIDED. Deterministic int8 value generator (no rand(), no clock),
// covering the full signed byte range including -128.
static int8_t detval(int i) {
    unsigned x = (unsigned)(i * 2654435761u + 12345u);
    x ^= x >> 13; x *= 2246822519u; x ^= x >> 16;
    return (int8_t)(x % 256u);
}

// FIXED driver. Do not edit. Four (M, N) cases with N a multiple of 16,
// deterministic A/B matrices, calls the learner's dot product, and
// prints each case's row-wise results.
int main() {
    struct Case { int M, N; };
    Case cases[] = {{4, 16}, {8, 32}, {1, 64}, {16, 128}};

    int seed_off = 0;
    for (const auto& c : cases) {
        std::vector<int8_t> A((size_t)c.M * c.N), B((size_t)c.M * c.N);
        for (int i = 0; i < c.M * c.N; ++i) {
            A[(size_t)i] = detval(seed_off + i);
            B[(size_t)i] = detval(seed_off + 1000000 + i);
        }
        seed_off += 100000;

        auto res = int8_widening_dot_product(A, B, c.M, c.N);
        printf("M=%d N=%d :", c.M, c.N);
        for (int32_t v : res) printf(" %d", v);
        printf("\n");
    }
    return 0;
}
