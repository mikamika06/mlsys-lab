// Reference: mark values that overflow fp16's max finite magnitude
// (65504) but are still within bf16's max finite magnitude
// (~3.3895314e38 -- bf16 keeps fp32's 8-bit exponent, just with fewer
// mantissa bits, so its dynamic range matches fp32 far beyond fp16).
__global__ void classify_overflow(const float* x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = fabsf(x[i]);
        float over_fp16 = v > 65504.0;
        float within_bf16 = v <= 3.3895314e38;
        out[i] = (over_fp16 > 0.0 && within_bf16 > 0.0) ? 1.0 : 0.0;
    }
}
