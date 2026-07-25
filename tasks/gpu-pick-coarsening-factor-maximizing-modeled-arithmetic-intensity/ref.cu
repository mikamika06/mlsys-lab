// Reference (single thread): thread coarsening has each thread compute C
// outputs instead of 1, reusing one shared load across all C of them.
// Registers per thread grow as base_regs + regs_per_c * C (one
// accumulator/address register set per extra output); arithmetic
// intensity AI(C) = flops_per_elem*C / (bytes_per_elem*(1+C)) grows
// monotonically with C (more reuse of the one shared load per extra
// output), so maximizing AI under a fixed register budget just means
// picking the LARGEST C the budget allows.
__global__ void coarsen_c(float* out, float reg_budget, float base_regs, float regs_per_c,
                            float flops_per_elem, float bytes_per_elem) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        float c = floorf((reg_budget - base_regs) / regs_per_c);
        float ai = (flops_per_elem * c) / (bytes_per_elem * (1.0f + c));
        out[0] = c;
        out[1] = ai;
    }
}
