// Reference: fused affine + relu. y[i] = relu(a*x[i] + b), one thread per
// element, ONE global read (x[i]) and ONE global write (y[i]) per thread
// -- the whole 2-stage chain (affine, then relu) computed in registers
// between them, never round-tripping the intermediate through memory.
__global__ void fused_affine_relu(float* y, const float* x, float a, float b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = a * x[i] + b;
        y[i] = v > 0.0f ? v : 0.0f;
    }
}
