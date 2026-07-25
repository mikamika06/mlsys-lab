#include "sol.hpp"

namespace {
constexpr int kCandidates[] = {16, 32, 64, 128, 256, 512, 1024};
constexpr int kNumCandidates = 7;
}  // namespace

void choose_kv_block_size(const int* token_sizes, int n, int table_overhead_per_block,
                           int* out_block_size, long* out_useful_bytes, long* out_allocated_bytes) {
    long useful = 0;
    for (int i = 0; i < n; i++) useful += token_sizes[i];

    int best_b = -1;
    long best_allocated = -1;
    for (int c = 0; c < kNumCandidates; c++) {
        int b = kCandidates[c];
        long allocated = 0;
        long total_blocks = 0;
        for (int i = 0; i < n; i++) {
            long blocks = (token_sizes[i] + b - 1) / b;
            allocated += blocks * b;
            total_blocks += blocks;
        }
        allocated += total_blocks * static_cast<long>(table_overhead_per_block);
        if (best_b == -1 || allocated < best_allocated) {
            best_b = b;
            best_allocated = allocated;
        }
    }
    *out_block_size = best_b;
    *out_useful_bytes = useful;
    *out_allocated_bytes = best_allocated;
}
