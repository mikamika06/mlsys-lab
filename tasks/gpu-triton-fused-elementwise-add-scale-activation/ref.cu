// Reference: fused elementwise add + scale + ReLU, one pass, one kernel
// -- no intermediate buffer for (a+b) or for the scaled sum ever touches
// global memory. out[i] = max(scale * (a[i] + b[i]), 0).
__global__ void fused_add_scale_relu(float* out, const float* a, const float* b, float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float sum = a[i] + b[i];
        float scaled = scale * sum;
        out[i] = fmaxf(scaled, 0.0f);
    }
}
