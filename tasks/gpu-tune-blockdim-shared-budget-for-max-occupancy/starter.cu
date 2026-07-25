// For each candidate blockDim block_dims[i], compute the modeled
// occupancy under fixed per-thread register/shared-memory usage and the
// SM's hardware limits, and write it to out[i]. See task.md for the
// exact 4-way resource-limit formula (threads, shared memory, registers,
// hardware max-blocks-per-SM). Use floorf() for every floor-division --
// this CUDA-C subset has no int/float cast operator.
__global__ void compute_occupancy(float* out, const float* block_dims, float regs_per_thread,
                                   float shared_bytes_per_thread, float max_threads_per_sm,
                                   float max_blocks_per_sm, float max_regs_per_sm,
                                   float max_shared_per_sm, int num_candidates) {
    int i = threadIdx.x;
    // TODO: compute occupancy for block_dims[i], guarded by i < num_candidates.
}
