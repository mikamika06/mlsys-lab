// Reference: modeled global-atomic-operation count for two histogram-
// update strategies over `n[i]` total updates spread across
// `block_size[i]`-thread blocks:
//   naive:      every one of the n updates does its OWN atomicAdd
//               straight into the (single, globally shared) output bin --
//               n atomic ops, every one of them serialized against every
//               other one hitting the same address.
//   privatized: each block accumulates its own updates into a PRIVATE
//               per-block counter first (ordinary shared-memory adds --
//               free contention, only threads in the same block ever
//               touch it), then exactly ONE thread per block does ONE
//               atomicAdd to flush that block's total into the global
//               bin -- only num_blocks atomic ops total, however large n
//               or block_size is.
__global__ void modeled_atomic_counts(float* naive_out, float* privatized_out,
                                       const float* n, const float* block_size, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        float total = n[i];
        float bs = block_size[i];
        float num_blocks = floorf((total + bs - 1.0f) / bs);
        naive_out[i] = total;
        privatized_out[i] = num_blocks;
    }
}
