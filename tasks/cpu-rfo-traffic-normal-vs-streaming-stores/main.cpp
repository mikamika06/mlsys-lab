// Fixed driver: pins the line size and runs a fixed table of buffer
// sizes (line-aligned and not) through both traffic functions. No
// timing, no randomness -- everything is deterministic integer
// arithmetic.
#include "sol.hpp"
#include <cstdio>

const int LINE_BYTES = 64;

namespace {
const long SIZES[] = {1, 64, 1024, 4096, 100000};
const int NUM_SIZES = sizeof(SIZES) / sizeof(SIZES[0]);
} // namespace

int main() {
    for (int i = 0; i < NUM_SIZES; i++) {
        long total_bytes = SIZES[i];
        long t = temporal_store_traffic(total_bytes);
        long nt = nontemporal_store_traffic(total_bytes);
        double ratio = static_cast<double>(t) / static_cast<double>(nt);
        printf("bytes=%ld temporal=%ld nontemporal=%ld ratio=%.6f\n", total_bytes, t, nt, ratio);
    }
    return 0;
}
