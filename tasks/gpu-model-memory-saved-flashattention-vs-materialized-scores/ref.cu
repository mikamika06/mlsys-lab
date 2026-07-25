// Reference: peak intermediate-memory ratio between materializing the
// full attention score matrix and FlashAttention's block-at-a-time
// computation. Materializing scores needs one seq_len x seq_len matrix
// (every query against every key) resident at once. FlashAttention only
// ever needs one block_size x block_size tile of scores at a time --
// each tile is folded into the running output and discarded before the
// next tile is computed, so its peak footprint never depends on
// seq_len, only on block_size.
__global__ void flash_vs_materialized_ratio(float* out, const float* seq_len, const float* block_size,
                                             const float* bytes_per_elem, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float materialized_bytes = seq_len[i] * seq_len[i] * bytes_per_elem[i];
        float flash_bytes = block_size[i] * block_size[i] * bytes_per_elem[i];
        out[i] = materialized_bytes / flash_bytes;
    }
}
