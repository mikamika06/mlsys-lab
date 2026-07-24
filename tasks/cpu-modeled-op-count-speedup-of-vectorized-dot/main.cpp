#include <cstdio>
#include "sol.hpp"

// FIXED driver. Every (n, width) pair is hand-picked (no rand()/time()) to
// cover exact multiples, non-multiples, and the n < width edge case, at
// widths matching real SIMD ISAs (NEON=4, AVX2=8, AVX-512=16).

namespace {
void run(const char* name, int n, int width) {
    double s = modeled_vector_speedup(n, width);
    printf("%s n=%d width=%d speedup=%.6f\n", name, n, width, s);
}
}  // namespace

int main() {
    run("neon_exact", 64, 4);
    run("neon_tail", 67, 4);
    run("avx2_exact", 64, 8);
    run("avx2_tail", 100, 8);
    run("avx512_exact", 256, 16);
    run("avx512_single", 1, 16);
    run("avx512_all_tail", 15, 16);
    return 0;
}
