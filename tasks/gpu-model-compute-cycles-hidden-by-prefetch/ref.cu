// Reference: total cycles for T pipeline tiles, each with load latency L
// and compute C, run serially vs. software-pipelined (prefetch overlaps
// tile i+1's load with tile i's compute).
//   serial(T,L,C)  = T * (L + C)
//   overlap(T,L,C) = L + (T-1) * max(L,C) + C
__global__ void pipeline_cycles(float* out_serial, float* out_overlap,
                                 const float* T, const float* L, const float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out_serial[i] = T[i] * (L[i] + C[i]);
        out_overlap[i] = L[i] + (T[i] - 1.0f) * fmaxf(L[i], C[i]) + C[i];
    }
}
