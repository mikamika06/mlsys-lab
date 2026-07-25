// Sum-reduce each 32-lane warp of `in` using __shfl_down_sync in a
// log-step ladder (delta = 16, 8, 4, 2, 1), no shared memory, no
// __syncthreads(). __shfl_down_sync must be the WHOLE right-hand side
// of its own assignment (e.g. `float got = __shfl_down_sync(mask, val, d);`),
// so read it into its own variable first, then add it into `val` in a
// separate statement. After all 5 steps, lane 0 of each warp holds that
// warp's total; write it to out[tid / 32].
__global__ void warp_reduce_sum(float* out, const float* in, int n) {
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp = tid / 32;
    float val = in[tid];
    // TODO: 5-step shfl_down ladder (delta 16,8,4,2,1), no guards needed.
    if (lane == 0) {
        out[warp] = val;
    }
}
