#include "sol.hpp"

// TODO: simulate the free-list allocator (first-fit ALLOC with
// splitting, FREE with two-sided coalescing) then return
// (total_free_bytes - largest_free_block_bytes) / total_free_bytes.
// See sol.hpp.
double external_fragmentation_ratio(long heap_bytes,
                                     const int* op_types, const int* op_sizes, const int* op_ids,
                                     int num_ops) {
    (void)heap_bytes; (void)op_types; (void)op_sizes; (void)op_ids; (void)num_ops;
    // your code here
    return 0.0;
}
