// For each configuration i, model the total number of GLOBAL atomic
// operations two histogram-update strategies need for n[i] updates spread
// across block_size[i]-thread blocks:
//   naive_out[i]:      one atomicAdd per update -> n[i].
//   privatized_out[i]: each block privately accumulates its updates, then
//                      ONE thread per block does ONE atomicAdd to flush
//                      the block's total -> the number of blocks needed
//                      to cover n[i] updates at block_size[i] threads
//                      each (round up).
__global__ void modeled_atomic_counts(float* naive_out, float* privatized_out,
                                       const float* n, const float* block_size, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        // TODO: naive_out[i] = n[i];
        // TODO: privatized_out[i] = ceil(n[i] / block_size[i]);
    }
}
