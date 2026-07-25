#include <cstring>
#include "sol.hpp"

namespace {

long run_ijk(int n) {
    cache_reset();
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++) {
                touch(a_addr(n, i, k));
                touch(b_addr(n, k, j));
                touch(c_addr(n, i, j));
            }
    return cache_misses();
}

long run_ikj(int n) {
    cache_reset();
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++)
            for (int j = 0; j < n; j++) {
                touch(a_addr(n, i, k));
                touch(b_addr(n, k, j));
                touch(c_addr(n, i, j));
            }
    return cache_misses();
}

long run_jki(int n) {
    cache_reset();
    for (int j = 0; j < n; j++)
        for (int k = 0; k < n; k++)
            for (int i = 0; i < n; i++) {
                touch(a_addr(n, i, k));
                touch(b_addr(n, k, j));
                touch(c_addr(n, i, j));
            }
    return cache_misses();
}

struct Entry {
    const char* name;
    long misses;
    int prio;
};

}  // namespace

void rank_matmul_orders(int n, char out[3][4]) {
    Entry entries[3] = {
        {"ijk", run_ijk(n), 0},
        {"ikj", run_ikj(n), 1},
        {"jki", run_jki(n), 2},
    };
    for (int i = 1; i < 3; i++) {
        Entry key = entries[i];
        int j = i - 1;
        while (j >= 0 && (entries[j].misses > key.misses ||
               (entries[j].misses == key.misses && entries[j].prio > key.prio))) {
            entries[j + 1] = entries[j];
            j--;
        }
        entries[j + 1] = key;
    }
    for (int i = 0; i < 3; i++) {
        std::strncpy(out[i], entries[i].name, 4);
    }
}
