// VALID but suboptimal: always uses C = 1 (no coarsening at all). Never
// exceeds the register budget, but leaves essentially all of the reuse
// (and therefore arithmetic intensity) the budget could have bought on
// the table.
__global__ void coarsen_c(float* out, float reg_budget, float base_regs, float regs_per_c,
                            float flops_per_elem, float bytes_per_elem) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        float c = 1.0f;
        float ai = (flops_per_elem * c) / (bytes_per_elem * (1.0f + c));
        out[0] = c;
        out[1] = ai;
    }
}
