// Fused add + scale + ReLU, one pass: out[i] = max(scale * (a[i] +
// b[i]), 0). Everything -- the add, the scale, the activation -- stays
// in registers for element i; nothing intermediate is written to
// global memory.
__global__ void fused_add_scale_relu(float* out, const float* a, const float* b, float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO: guard i < n, then out[i] = fmaxf(scale * (a[i] + b[i]), 0.0f);
}
