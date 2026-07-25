// Compute y[i] = relu(a*x[i] + b) for i in [0, n), one thread per
// element. Fuse the affine step and the relu step into ONE pass: read
// x[i] once, write y[i] once, no intermediate round-trip through global
// memory.
__global__ void fused_affine_relu(float* y, const float* x, float a, float b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO: guard i < n, then y[i] = relu(a * x[i] + b);
}
