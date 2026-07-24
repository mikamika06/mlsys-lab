#pragma once
#include <cstdint>

// One table slot. Under the LP64 ABI: bool(1) + 7 bytes padding (long needs
// 8-byte alignment) + long(8) + long(8) = 24 bytes.
struct Slot {
    bool occupied;
    long long key;
    long long value;
};

// Fixed 64-bit multiplicative (Fibonacci) hash, computed with unsigned
// wraparound (well-defined, unlike signed overflow).
inline uint64_t hash_key(long long k) {
    return (uint64_t)k * 11400714819323198485ULL;
}

// Insert key `k` into `table` (an array of `C` slots, all initially
// !occupied) using open addressing with LINEAR PROBING:
//   1. i = hash_key(k) % C
//   2. while table[i].occupied: i = (i + 1) % C
//   3. mark table[i] occupied, store key = k, value = k * 2
// Return the slot index used.
int insert_probe(Slot* table, int C, long long k);
