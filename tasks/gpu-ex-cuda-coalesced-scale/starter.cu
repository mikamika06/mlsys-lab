// Compute out[i] = s * in[i] for i in [0, n), one element per thread.
// Keep the access coalesced: thread i must touch address i in both
// `in` and `out` (no stride, no permutation).
__global__ void scale(float* out, const float* in, int n, float s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO: guard i < n, then out[i] = s * in[i];
}
