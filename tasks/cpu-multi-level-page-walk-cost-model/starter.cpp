#include "sol.hpp"

// TODO: maintain 4 independent LRU caches (one per page-table level, of
// capacity cap[i]) and sum hit_cycles/miss_cycles across the 4 levels
// plus data_cycles, for every address -- see sol.hpp.
long page_walk_cycles(const int* keys, int num_addrs, const int* cap,
                       long hit_cycles, long miss_cycles, long data_cycles) {
    (void)keys; (void)num_addrs; (void)cap; (void)hit_cycles; (void)miss_cycles; (void)data_cycles;
    // your code here
    return 0;
}
