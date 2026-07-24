#include <cstdio>
#include <cstring>
#include "sol.hpp"

static const int ALLOC = 0, FREE = 1;

// FIXED scenario: 6 blocks of assorted sizes; 6 initial allocations fill
// the whole heap, then 3 frees carve out a fragmented set of free blocks
// (sizes 64, 32, 16 -- sum 112, largest 64), then a mix of requests that
// succeed, and requests (100, then later 40) that fail purely because no
// single free block is large enough, even though the free bytes sum to
// more than the request.
int main() {
    const int block_sizes[6] = {64, 128, 32, 256, 16, 64};
    const int num_blocks = 6;

    // op_ids: for a FREE op, the op-index of the ALLOC it releases.
    const int op_types[17] = {
        ALLOC, ALLOC, ALLOC, ALLOC, ALLOC, ALLOC,   // 0..5: fill the heap
        FREE, FREE, FREE,                            // 6..8: free blocks 0,2,4
        ALLOC,                                        // 9: 100 -- fragmentation FAIL
        ALLOC,                                        // 10: 50 -- succeeds (block0)
        ALLOC,                                        // 11: 40 -- fragmentation FAIL
        FREE,                                          // 12: free block used by op 5
        ALLOC, ALLOC, ALLOC,                          // 13..15: 60, 20, 16 -- all succeed
        ALLOC,                                         // 16: 1 -- heap full, FAIL
    };
    const int op_sizes[17] = {
        64, 128, 32, 256, 16, 64,
        0, 0, 0,
        100,
        50,
        40,
        0,
        60, 20, 16,
        1,
    };
    const int op_ids[17] = {
        -1, -1, -1, -1, -1, -1,
        0, 2, 4,
        -1,
        -1,
        -1,
        5,
        -1, -1, -1,
        -1,
    };
    const int num_ops = 17;

    int out_labels[17];
    memset(out_labels, 0, sizeof(out_labels));

    classify_allocations(block_sizes, num_blocks, op_types, op_sizes, op_ids, num_ops, out_labels);

    printf("labels=");
    bool first = true;
    for (int i = 0; i < num_ops; i++) {
        if (op_types[i] != ALLOC) continue;
        if (!first) printf(",");
        printf("%d", out_labels[i]);
        first = false;
    }
    printf("\n");
    return 0;
}
