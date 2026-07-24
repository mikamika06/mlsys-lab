#include "sol.hpp"

// Reference: textbook Belady OPT simulation. `resident[k]` holds the page
// id in cache slot k, or -1 if the slot is empty.
int belady_opt_misses(const int* refs, int n, int capacity) {
    int* resident = new int[capacity];
    for (int k = 0; k < capacity; k++) resident[k] = -1;

    int misses = 0;
    for (int i = 0; i < n; i++) {
        int page = refs[i];

        // hit?
        int slot = -1;
        for (int k = 0; k < capacity; k++) {
            if (resident[k] == page) {
                slot = k;
                break;
            }
        }
        if (slot != -1) continue;  // hit: no eviction, no miss

        misses++;

        // find an empty slot first
        int target = -1;
        for (int k = 0; k < capacity; k++) {
            if (resident[k] == -1) {
                target = k;
                break;
            }
        }

        if (target == -1) {
            // cache full: evict whichever resident page is used furthest
            // in the future (or never again).
            int worst_slot = 0;
            int worst_next_use = -1;
            for (int k = 0; k < capacity; k++) {
                int next_use = n;  // "never used again" sorts as furthest
                for (int j = i + 1; j < n; j++) {
                    if (refs[j] == resident[k]) {
                        next_use = j;
                        break;
                    }
                }
                if (next_use > worst_next_use) {
                    worst_next_use = next_use;
                    worst_slot = k;
                }
            }
            target = worst_slot;
        }

        resident[target] = page;
    }

    delete[] resident;
    return misses;
}
