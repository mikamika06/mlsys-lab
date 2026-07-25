// Compute out[i] = s * in[i] for i in [0, n), and work correctly no matter
// how small the launched grid is relative to n -- if
// blockDim.x * gridDim.x < n, a single one-element-per-thread pass leaves
// most of the array untouched. Loop each thread forward by the total
// thread count (blockDim.x * gridDim.x) until it covers every index.
__global__ void scale_grid_stride(float* out, const float* in, int n, float s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // BUG: only ever handles ONE element per thread -- correct only when
    // blockDim.x * gridDim.x >= n. Turn this into a grid-stride loop.
    if (i < n) {
        out[i] = s * in[i];
    }
}
