// TODO: compute the total A loads WITH register blocking -- M*(N/C)
// threads (each of the M rows split into N/C column-groups of C
// outputs each), each thread loading its A row K times (once per k,
// reused across its C outputs) -- and the total B loads, which register
// blocking does NOT reduce (every one of the M*N output elements needs
// its own K fresh loads from B, coarsening or not). See ref.cu.
__global__ void derive_loads(int M, int N, int K, int C, float* out) {
    out[0] = 0.0;
    out[1] = 0.0;
}
