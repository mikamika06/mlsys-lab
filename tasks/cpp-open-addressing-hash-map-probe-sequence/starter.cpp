#include "sol.hpp"

// TODO: implement linear probing as described in sol.hpp. Right now this
// always "inserts" at slot 0 without checking occupancy or advancing, so
// every key after the first collides silently and every returned slot
// index is wrong except possibly the very first key.
int insert_probe(Slot* table, int C, long long k) {
    (void)table; (void)C; (void)k;
    return 0;  // your code here
}
