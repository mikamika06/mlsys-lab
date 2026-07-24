// Three ways to scale n elements of g by a, each with a different access
// pattern:
//   unit_stride     -- thread idx touches g[idx]:            fully coalesced
//   reversed_stride -- thread idx touches g[n-1-idx]:         STILL fully
//                      coalesced -- a warp's 32 lanes still touch 32
//                      CONSECUTIVE addresses, just in reverse order; the
//                      hardware coalesces by which SEGMENT a warp touches,
//                      not by which lane maps to which address in it.
//   stride4         -- thread idx touches g[idx*4]:           NOT coalesced
//                      -- a warp's 32 lanes now span 128 elements, several
//                      separate 128-byte segments instead of one.
__global__ void unit_stride(float* g, float a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) g[idx] = a * g[idx];
}

__global__ void reversed_stride(float* g, float a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int j = n - 1 - idx;
        g[j] = a * g[j];
    }
}

__global__ void stride4(float* g, float a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int j = idx * 4;
        g[j] = a * g[j];
    }
}
