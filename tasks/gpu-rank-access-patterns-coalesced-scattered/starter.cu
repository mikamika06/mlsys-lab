// Implement all three kernels: each scales n elements of g by a, but at a
// DIFFERENT address for thread idx:
//   unit_stride:     g[idx]
//   reversed_stride: g[n - 1 - idx]
//   stride4:         g[idx * 4]
// Guard every access with idx < n.

__global__ void unit_stride(float* g, float a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO: g[idx] = a * g[idx];
}

__global__ void reversed_stride(float* g, float a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO: g[n - 1 - idx] = a * g[n - 1 - idx];
}

__global__ void stride4(float* g, float a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO: g[idx * 4] = a * g[idx * 4];
}
