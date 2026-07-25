// Compute the tail-waste-free block count: total_blocks =
// ceil(N/block_size); if total_blocks > max_concurrent, the answer is
// the LARGEST multiple of max_concurrent that is <= total_blocks
// (integer-divide then multiply back); otherwise (a single wave
// regardless) the answer is just total_blocks. Write it to out[idx].
__global__ void optimal_grid_blocks(float* out, int idx, int N, int block_size, int max_concurrent) {
    int total_blocks = (N + block_size - 1) / block_size;
    // your code here
}
