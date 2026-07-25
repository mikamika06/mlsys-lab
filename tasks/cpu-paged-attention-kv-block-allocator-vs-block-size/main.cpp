#include <cstdio>
#include "sol.hpp"

// FIXED driver: a deterministic 30-token workload (no rand()/time()) with
// KV entry sizes spread across 40..259 bytes, and a 24-byte per-block
// table overhead.
int main() {
    const int n = 30;
    int token_sizes[n];
    for (int i = 0; i < n; i++) token_sizes[i] = 40 + (i * 53) % 220;
    const int table_overhead_per_block = 24;

    int block_size = -1;
    long useful = -1, allocated = -1;
    choose_kv_block_size(token_sizes, n, table_overhead_per_block, &block_size, &useful, &allocated);

    printf("block_size=%d useful_bytes=%ld allocated_bytes=%ld\n", block_size, useful, allocated);
    return 0;
}
