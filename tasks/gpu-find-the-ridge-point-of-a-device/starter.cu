// ridge point (FLOP/byte) = peak_flops[i] / peak_bw[i].
__global__ void ridge_point(float* out, const float* peak_flops, const float* peak_bw, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO: guard i < n, then out[i] = peak_flops[i] / peak_bw[i];
}
