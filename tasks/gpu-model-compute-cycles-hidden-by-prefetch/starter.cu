// serial(T,L,C) = T * (L + C).
// overlap(T,L,C) = L + (T-1) * max(L,C) + C.
__global__ void pipeline_cycles(float* out_serial, float* out_overlap,
                                 const float* T, const float* L, const float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO: guard i < n, then compute out_serial[i] and out_overlap[i]
    // per the formulas above.
}
