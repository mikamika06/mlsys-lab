#include "sol.hpp"

// TODO: implement the three allocators described in sol.hpp (address-ordered
// first-fit, address-ordered best-fit, power-of-two buddy), replay the trace
// through each of them independently, and write their final external
// fragmentation (total_free_bytes - largest_contiguous_free_block_bytes)
// into out[0..3).
void fragmentation_after_trace(const int* op_kind, const int* op_arg, int num_ops, int* out) {
    (void)op_kind; (void)op_arg; (void)num_ops; (void)out;
    // your code here
}
