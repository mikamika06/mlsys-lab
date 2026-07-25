// BUG: each thread computes ONE element and stops -- there is no stride
// loop. When n is bigger than gridDim.x * blockDim.x (the total number
// of launched threads), every element from gridDim.x * blockDim.x up to
// n - 1 is a "tail" that no thread ever touches.
__global__ void grid_stride_scale(float* out, const float* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = 2.0f * in[i] + 1.0f;
    }
}
