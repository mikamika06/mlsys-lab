// TODO: for each i, let v = fabsf(x[i]). Set out[i] = 1.0 if v overflows
// fp16's max finite magnitude (65504) but is still within bf16's max
// finite magnitude (~3.3895314e38); otherwise out[i] = 0.0.
__global__ void classify_overflow(const float* x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = 0.0;
    }
}
