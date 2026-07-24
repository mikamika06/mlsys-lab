#include <cstdint>
#include <cstdio>

#include "sol.hpp"

Tracked operator+(Tracked a, Tracked b) {
    Tracked r;
    r.value = a.value + b.value;
    r.depth = (a.depth > b.depth ? a.depth : b.depth) + 1;
    return r;
}

// FIXED driver: N = 4096 deterministic integer-valued doubles (so the sum
// is exact regardless of accumulation order), reduced with K = 4
// accumulators. Prints the sum, and a depth_ok flag that's 1 only if the
// resulting critical-path depth stayed well below what a single serial
// accumulator would produce (~4096) -- i.e. the accumulators were actually
// used independently, not just declared and ignored.
int main() {
    const int N = 4096;
    const int K = 4;
    const int DEPTH_THRESHOLD = 1200;  // real 4-accumulator depth is ~1027; single-acc is ~4096

    static Tracked x[N];
    for (int i = 0; i < N; i++) {
        uint32_t h = (uint32_t)i * 1103515245u + 12345u;
        x[i].value = (double)((int)((h >> 8) & 1023u) - 512);  // integer in [-512, 511]
        x[i].depth = 0;
    }

    Tracked total = reduce_with_accumulators(x, N, K);

    printf("%.6f\n", total.value);
    printf("depth_ok %d\n", total.depth <= DEPTH_THRESHOLD ? 1 : 0);
    return 0;
}
