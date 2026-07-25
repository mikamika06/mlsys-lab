#include "sol.hpp"

// TODO: for each candidate block size b in {16,32,64,128,256,512,1024},
// compute allocated(b) = sum(ceil(s/b)*b) + total_blocks(b)*table_overhead
// (see sol.hpp), and keep the candidate with the SMALLEST allocated(b)
// (ties -> smaller b). useful_bytes is just sum(token_sizes).
void choose_kv_block_size(const int* token_sizes, int n, int table_overhead_per_block,
                           int* out_block_size, long* out_useful_bytes, long* out_allocated_bytes) {
    (void)token_sizes; (void)n; (void)table_overhead_per_block;
    // your code here
    *out_block_size = 0;
    *out_useful_bytes = 0;
    *out_allocated_bytes = 0;
}
