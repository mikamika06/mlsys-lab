// For each scenario i, compute the ratio of peak intermediate bytes:
// (seq_len[i]^2 * bytes_per_elem[i]) for materializing the full
// attention score matrix, divided by (block_size[i]^2 * bytes_per_elem[i])
// for FlashAttention's one-tile-at-a-time computation. Write the ratio
// to out[i]. See task.md.
__global__ void flash_vs_materialized_ratio(float* out, const float* seq_len, const float* block_size,
                                             const float* bytes_per_elem, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO: out[i] = (seq_len[i]^2 * bytes_per_elem[i]) / (block_size[i]^2 * bytes_per_elem[i]), guarded by i < n.
}
