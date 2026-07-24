// Fixed driver: pins the allocator model constants and runs a fixed
// table of (count, obj_bytes) cases through all three functions. No
// timing, no real malloc() calls -- everything is deterministic integer
// arithmetic over a fixed model.
#include "sol.hpp"
#include <cstdio>

const int HEADER_BYTES = 16;
const int ALIGN_BYTES = 16;

namespace {
struct Case { int count, obj_bytes; };

const Case CASES[] = {
    {1000, 8},     // many tiny objects: header dominates badly
    {1000, 16},
    {1000, 64},
    {1000, 256},   // larger objects: relative overhead shrinks
    {1, 1000},     // a single object: pool buys nothing
    {10000, 4},    // extreme: very many very tiny objects
};
const int NUM_CASES = sizeof(CASES) / sizeof(CASES[0]);
} // namespace

int main() {
    for (int i = 0; i < NUM_CASES; i++) {
        const Case& c = CASES[i];
        long m = malloc_per_object_footprint(c.count, c.obj_bytes);
        long p = pool_footprint(c.count, c.obj_bytes);
        double r = footprint_ratio(c.count, c.obj_bytes);
        printf("count=%d obj=%d malloc=%ld pool=%ld ratio=%.6f\n", c.count, c.obj_bytes, m, p, r);
    }
    return 0;
}
