#include <cstdio>
#include <vector>
#include "sol.hpp"

int main() {
    // A structured walk over virtual addresses, nested like real paging:
    // level0 (PML4) changes rarely, level3 (PT) changes on every address.
    const int NA = 2, NB = 3, NC = 4, ND = 5;  // 2*3*4*5 = 120 addresses
    const int NUM_ADDRS = NA * NB * NC * ND;

    std::vector<int> keys(NUM_ADDRS * 4);
    int j = 0;
    for (int a = 0; a < NA; a++)
        for (int b = 0; b < NB; b++)
            for (int c = 0; c < NC; c++)
                for (int d = 0; d < ND; d++) {
                    keys[j * 4 + 0] = a;
                    keys[j * 4 + 1] = b;
                    keys[j * 4 + 2] = c;
                    keys[j * 4 + 3] = d;
                    j++;
                }

    // Deliberately tight capacities (smaller than each level's distinct
    // key count) so real evictions -- not just cold misses -- occur.
    const int cap[4] = {1, 2, 2, 3};
    const long HIT = 4, MISS = 120, DATA = 4;

    long total = page_walk_cycles(keys.data(), NUM_ADDRS, cap, HIT, MISS, DATA);
    printf("total_cycles=%ld\n", total);
    return 0;
}
