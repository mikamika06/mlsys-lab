// attainable FLOP/s = min(peak_flops[i], ai[i] * peak_bw[i]).
__global__ void attainable_flops(float* out, const float* peak_flops, const float* peak_bw,
                                  const float* ai, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO: guard i < n, then out[i] = fminf(peak_flops[i], ai[i] * peak_bw[i]);
}
