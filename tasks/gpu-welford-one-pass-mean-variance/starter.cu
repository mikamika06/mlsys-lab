// Implement single-thread (lane 0 only) Welford one-pass mean/variance.
// Read x[0..n), write out[0] = mean, out[1] = population variance
// (M2 / n). Only the lane with gid == 0 should do any work.
__global__ void welford_kernel(const float* x, float* out, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid != 0) return;
    // TODO: Welford's one-pass mean/variance over x[0..n).
    out[0] = 0.0;
    out[1] = 0.0;
}
