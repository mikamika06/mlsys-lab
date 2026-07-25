// Elementwise scale: out[i] = s * in[i] for i in [0, n). n is NOT a
// multiple of blockDim.x * gridDim.x -- the last block has some threads
// whose global index i is >= n, and they must not read or write anything.
//
// BUG: the mask below checks i against the total number of threads the
// launch started (blockDim.x * gridDim.x), which is always true by
// construction -- it never actually excludes the tail threads of the last
// block. Fix it to check against the real data size, n, instead.
__global__ void scale_masked(float* out, const float* in, int n, float s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < blockDim.x * gridDim.x) {
        out[i] = s * in[i];
    }
}
