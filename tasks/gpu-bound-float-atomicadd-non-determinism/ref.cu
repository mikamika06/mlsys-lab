// Reference: standard block-level tree (binary) reduction. Every thread
// loads one value into shared memory; then, with the active stride
// halving each round (32, 16, 8, ... 1), thread `tid` (for tid < stride)
// folds s[tid + stride] into s[tid]. Thread 0 writes the final sum.
// This is what a real GPU reduction kernel looks like when it avoids
// atomicAdd -- but the COMBINING ORDER still depends entirely on which
// value started out at which shared-memory slot, i.e. on the caller's
// input ordering, exactly as a race among concurrent atomicAdd()s would
// make the combining order depend on scheduling.
__global__ void block_reduce_sum(const float* x, float* out, int n) {
    __shared__ float s[64];
    int tid = threadIdx.x;
    s[tid] = x[tid];
    __syncthreads();

    int stride = blockDim.x / 2;
    while (stride > 0) {
        if (tid < stride) {
            s[tid] = s[tid] + s[tid + stride];
        }
        __syncthreads();
        stride = stride / 2;
    }

    if (tid == 0) {
        out[0] = s[0];
    }
}
