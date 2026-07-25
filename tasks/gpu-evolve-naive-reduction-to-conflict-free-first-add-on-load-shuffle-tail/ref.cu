// Reference: Harris's optimized reduction ladder, all 3 rungs.
//   1) first-add-on-load: only the first 128 (of 256 launched) threads do
//      any work; each one sums TWO elements (x[tid], x[tid+128]) into
//      shared memory before the tree even starts, halving the tree depth.
//   2) conflict-free (sequential) addressing: stride HALVES (128->64->32)
//      with a uniform "tid < stride" condition -- every active thread's
//      index is contiguous, no bank conflicts, no warp divergence beyond
//      the single active/inactive split.
//   3) shuffle tail: once down to 32 live values (one warp), finish the
//      last 5 halvings with __shfl_down_sync instead of shared memory +
//      __syncthreads() -- a warp is already synchronous.
__global__ void reduce_sum(float* out, const float* x, int n) {
    __shared__ float sdata[128];
    int tid = threadIdx.x;
    if (tid < 128) {
        sdata[tid] = x[tid] + x[tid + 128];
    }
    __syncthreads();

    if (tid < 64) { sdata[tid] += sdata[tid + 64]; }
    __syncthreads();
    if (tid < 32) { sdata[tid] += sdata[tid + 32]; }
    __syncthreads();

    if (tid < 32) {
        float val = sdata[tid];
        val += __shfl_down_sync(0xffffffff, val, 16);
        val += __shfl_down_sync(0xffffffff, val, 8);
        val += __shfl_down_sync(0xffffffff, val, 4);
        val += __shfl_down_sync(0xffffffff, val, 2);
        val += __shfl_down_sync(0xffffffff, val, 1);
        if (tid == 0) {
            out[0] = val;
        }
    }
}
