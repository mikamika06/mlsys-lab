#include <cstdio>
#include "sol.hpp"

static const int ALLOC = 0, FREE = 1;

int main() {
    // --- Scenario A: 1000-byte heap, 5 allocations exactly fill it,
    // then 3 non-adjacent allocations are freed (0, 2, 4), leaving three
    // separate free blocks -- no coalescing opportunity, since none of
    // the freed blocks end up physically adjacent to each other.
    const long heapA = 1000;
    const int typesA[8]  = {ALLOC, ALLOC, ALLOC, ALLOC, ALLOC, FREE, FREE, FREE};
    const int sizesA[8]  = {100,   200,   150,   250,   300,   0,    0,    0};
    const int idsA[8]    = {-1,    -1,    -1,    -1,    -1,    0,    2,    4};
    double ratioA = external_fragmentation_ratio(heapA, typesA, sizesA, idsA, 8);

    // --- Scenario B: 500-byte heap. Fill it with 5 allocations. Frees
    // of ops 1 and 2 are mutually adjacent, so they coalesce into one
    // run; the free of op 4 is separate (op 3's block stays used between
    // them), leaving two free blocks; a final small allocation then
    // splits the bigger of the two, leaving a mixed free-list to compute
    // the ratio over.
    const long heapB = 500;
    const int typesB[9] = {ALLOC, ALLOC, ALLOC, ALLOC, ALLOC, FREE, FREE, FREE, ALLOC};
    const int sizesB[9] = {50,    80,    70,    60,    240,   0,    0,    0,    40};
    const int idsB[9]   = {-1,    -1,    -1,    -1,    -1,    1,    2,    4,    -1};
    double ratioB = external_fragmentation_ratio(heapB, typesB, sizesB, idsB, 9);

    printf("ratioA=%.6f ratioB=%.6f\n", ratioA, ratioB);
    return 0;
}
