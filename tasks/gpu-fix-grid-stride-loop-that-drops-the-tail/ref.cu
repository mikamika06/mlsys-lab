// Reference: a proper grid-stride loop. When n is bigger than the total
// number of launched threads (gridDim.x * blockDim.x), thread i handles
// element i, then i + stride, then i + 2*stride, ... until it runs past
// n -- every element gets covered no matter how small the launch grid is
// relative to n.
__global__ void grid_stride_scale(float* out, const float* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        out[i] = 2.0f * in[i] + 1.0f;
    }
}
