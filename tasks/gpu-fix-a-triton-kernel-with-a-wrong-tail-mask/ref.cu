// Reference: elementwise scale with a correct tail mask. n is not a
// multiple of blockDim.x * gridDim.x, so the last block launches some
// threads whose global index i is >= n -- they must not touch memory at
// all. The mask compares i against n (the real data size), not against
// the number of threads the launch happens to have started.
__global__ void scale_masked(float* out, const float* in, int n, float s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = s * in[i];
    }
}
