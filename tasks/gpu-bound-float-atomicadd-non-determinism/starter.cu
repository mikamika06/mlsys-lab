// TODO: block-level tree reduction. Load x[tid] into shared memory
// s[tid], syncthreads, then repeatedly halve `stride` (starting at
// blockDim.x / 2): while stride > 0, threads with tid < stride do
// s[tid] += s[tid + stride], syncthreads after each round, until
// stride reaches 0. Thread 0 writes out[0] = s[0].
__global__ void block_reduce_sum(const float* x, float* out, int n) {
    __shared__ float s[64];
    int tid = threadIdx.x;
    if (tid == 0) {
        out[0] = 0.0;
    }
}
