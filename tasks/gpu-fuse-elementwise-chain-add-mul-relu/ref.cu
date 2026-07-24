// Fused elementwise chain: out[i] = max(0, (a[i] + b[i]) * c[i]).
// Every intermediate value stays in a register (a local scalar) -- each
// thread reads a[i], b[i], c[i] from global memory EXACTLY ONCE and writes
// out[i] EXACTLY ONCE, instead of bouncing intermediates through `out`.
__global__ void fuse(float* out, const float* a, const float* b, const float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float sum = a[i] + b[i];
        float prod = sum * c[i];
        out[i] = max(prod, 0.0f);
    }
}
