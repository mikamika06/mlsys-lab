#include "sol.hpp"

void classify_allocations(const int* block_sizes, int num_blocks,
                           const int* op_types, const int* op_sizes, const int* op_ids,
                           int num_ops, int* out_labels) {
    bool used[64] = {false};  // num_blocks is always small in this task's scenarios
    int alloc_block[4096];    // alloc_block[i] = block index that ALLOC op i used, or -1
    for (int i = 0; i < num_ops; i++) alloc_block[i] = -1;

    for (int i = 0; i < num_ops; i++) {
        if (op_types[i] == 0) {  // ALLOC
            int req = op_sizes[i];
            int found = -1;
            for (int b = 0; b < num_blocks; b++) {
                if (!used[b] && block_sizes[b] >= req) { found = b; break; }
            }
            if (found >= 0) {
                used[found] = true;
                alloc_block[i] = found;
                out_labels[i] = 1;
            } else {
                out_labels[i] = 0;
            }
        } else {  // FREE
            int aid = op_ids[i];
            int b = (aid >= 0) ? alloc_block[aid] : -1;
            if (b >= 0) used[b] = false;
        }
    }
}
