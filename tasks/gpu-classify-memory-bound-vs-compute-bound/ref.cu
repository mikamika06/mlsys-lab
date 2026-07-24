// Reference: roofline classification, one element per thread, coalesced.
// Thread i reads ai[i] and writes out[i] -> consecutive addresses within a
// warp -> coalesces into one 128-byte transaction per access step.
__global__ void classify(float* out, const float* ai, int n, float ridge) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = (ai[i] >= ridge) ? 1.0f : 0.0f;
    }
}
