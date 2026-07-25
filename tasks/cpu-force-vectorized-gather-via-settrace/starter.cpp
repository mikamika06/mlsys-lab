#include "sol.hpp"

// TODO: keep your OWN "seen this index value before" + "cached value"
// arrays (size table_len), and only touch()/read table[idx] the FIRST
// time a given idx shows up. See sol.hpp.
void gather_dedup(const float* table, int table_len, const int* indices,
                   int n, float* output) {
    (void)table_len;
    // your code here
    for (int i = 0; i < n; i++) {
        int idx = indices[i];
        touch(table_addr(idx));
        output[i] = table[idx];
    }
}
