// Reference: modeled occupancy of a kernel at each candidate blockDim,
// given fixed per-thread register and shared-memory usage and an SM's
// hardware limits. How many blocks of `bd` threads can be resident on
// one SM at once is capped by FOUR independent budgets -- threads,
// shared memory, registers, and the hardware's own max-blocks-per-SM --
// and the SM can never exceed the TIGHTEST of the four:
//
//   blocks_by_threads = floor(max_threads_per_sm / bd)
//   blocks_by_shared  = floor(max_shared_per_sm / (shared_bytes_per_thread * bd))
//   blocks_by_regs    = floor(max_regs_per_sm / (regs_per_thread * bd))
//   actual_blocks     = min(blocks_by_threads, blocks_by_shared, blocks_by_regs, max_blocks_per_sm)
//   occupancy         = (actual_blocks * bd) / max_threads_per_sm
//
// This CUDA-C subset has no integer/float distinction in its own right
// (every gmem value is already a plain number) and no C-style casts, so
// "floor divide" is spelled out explicitly with floorf().
__global__ void compute_occupancy(float* out, const float* block_dims, float regs_per_thread,
                                   float shared_bytes_per_thread, float max_threads_per_sm,
                                   float max_blocks_per_sm, float max_regs_per_sm,
                                   float max_shared_per_sm, int num_candidates) {
    int i = threadIdx.x;
    if (i < num_candidates) {
        float bd = block_dims[i];
        float blocks_by_threads = floorf(max_threads_per_sm / bd);
        float shared_per_block = shared_bytes_per_thread * bd;
        float blocks_by_shared = floorf(max_shared_per_sm / shared_per_block);
        float regs_per_block = regs_per_thread * bd;
        float blocks_by_regs = floorf(max_regs_per_sm / regs_per_block);
        float actual_blocks = fminf(fminf(blocks_by_threads, blocks_by_shared),
                                     fminf(blocks_by_regs, max_blocks_per_sm));
        float resident_threads = actual_blocks * bd;
        out[i] = resident_threads / max_threads_per_sm;
    }
}
