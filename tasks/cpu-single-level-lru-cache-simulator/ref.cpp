#include "sol.hpp"

HitMiss simulate_lru(const long* addrs, int n, int capacity, int line_bytes) {
    static const int MAX_CAPACITY = 64;
    long resident[MAX_CAPACITY]; // index 0 = LRU, last used index = MRU
    int count = 0;

    long hits = 0, misses = 0;
    for (int i = 0; i < n; i++) {
        long line = addrs[i] / line_bytes;

        int found = -1;
        for (int k = 0; k < count; k++) {
            if (resident[k] == line) {
                found = k;
                break;
            }
        }

        if (found != -1) {
            hits++;
            for (int k = found; k < count - 1; k++) resident[k] = resident[k + 1];
            resident[count - 1] = line;
        } else {
            misses++;
            if (count == capacity) {
                for (int k = 0; k < count - 1; k++) resident[k] = resident[k + 1];
                resident[count - 1] = line;
            } else {
                resident[count] = line;
                count++;
            }
        }
    }

    return HitMiss{hits, misses};
}
