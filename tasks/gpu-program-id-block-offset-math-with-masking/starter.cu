// n is not a multiple of blockDim.x * gridDim.x -- the last block has
// threads with i >= n. `out` is sized to the FULL launch (padded past n),
// so every thread stores to it unconditionally; `in` only has n real
// elements, so the LOAD needs a mask: read in[i] when i < n, otherwise use
// 0.0 instead of running off the end of `in`.
__global__ void masked_scale_fill(float* out, const float* in, int n, float s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO: v = (i < n) ? in[i] : 0.0f;  out[i] = s * v;
}
