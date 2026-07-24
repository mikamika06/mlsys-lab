#include "sol.hpp"

int insert_probe(Slot* table, int C, long long k) {
    int i = (int)(hash_key(k) % (uint64_t)C);
    while (table[i].occupied) {
        i = (i + 1) % C;
    }
    table[i].occupied = true;
    table[i].key = k;
    table[i].value = k * 2;
    return i;
}
