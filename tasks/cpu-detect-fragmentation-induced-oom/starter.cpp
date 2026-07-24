#include "sol.hpp"

// TODO: simulate the fixed-block heap: first-fit ALLOC over FREE blocks
// big enough for the request, FREE toggles a block back to free -- see
// sol.hpp. `out_labels` starts zero-filled by the driver.
void classify_allocations(const int* block_sizes, int num_blocks,
                           const int* op_types, const int* op_sizes, const int* op_ids,
                           int num_ops, int* out_labels) {
    (void)block_sizes; (void)num_blocks; (void)op_types; (void)op_sizes; (void)op_ids; (void)num_ops;
    (void)out_labels;
    // your code here
}
