// Reference: attainable FLOP/s from the roofline model.
// attainable = min(peak_flops, arithmetic_intensity * peak_bw).
__global__ void attainable_flops(float* out, const float* peak_flops, const float* peak_bw,
                                  const float* ai, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = fminf(peak_flops[i], ai[i] * peak_bw[i]);
    }
}
