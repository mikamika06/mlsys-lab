// Reference: inverted dropout. mask[i] is a precomputed keep-mask (1.0 =
// keep, 0.0 = drop). Kept elements are rescaled by 1/(1-p) so the expected
// value of each element is unchanged; dropped elements become exactly 0.
// One element per thread, coalesced (thread i touches address i everywhere).
__global__ void dropout(float* out, const float* x, const float* mask, int n, float p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = mask[i] * x[i] / (1.0f - p);
    }
}
