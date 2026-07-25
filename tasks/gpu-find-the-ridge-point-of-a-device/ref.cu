// Reference: ridge point (FLOP/byte) = peak_flops / peak_bw.
__global__ void ridge_point(float* out, const float* peak_flops, const float* peak_bw, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = peak_flops[i] / peak_bw[i];
    }
}
