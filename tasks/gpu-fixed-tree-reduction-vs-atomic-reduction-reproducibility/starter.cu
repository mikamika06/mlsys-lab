// tree_reduce_sum: sum x[0..n) into out[0] with a sequential-addressing
// tree reduction in shared memory (stride = blockDim.x/2 down to 1,
// halving each step; tid < stride active; barrier every step).
__global__ void tree_reduce_sum(float* out, const float* x, int n) {
    __shared__ float sdata[32];
    int tid = threadIdx.x;
    // TODO: load, barrier, sequential-addressing reduction loop, barrier
    // each step, then out[0] = sdata[0] from thread 0.
}

// ordered_reduce_sum: single-threaded, sum x[order[0]], x[order[1]], ...,
// x[order[n-1]] into out[0], strictly in that sequence (one add at a
// time, in `order`'s order -- not sorted, not reduced in a tree).
__global__ void ordered_reduce_sum(float* out, const float* x, const float* order, int n) {
    // TODO: if (threadIdx.x == 0), accumulate x[(int)order[k]] for
    // k = 0..n-1 in order, then out[0] = acc.
}
