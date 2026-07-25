// Reduce 32 values, one per lane (n=32, single warp), down to a single
// total in out[0]. Use ONLY __shfl_down_sync -- no __shared__ array, no
// __syncthreads(). __shfl_down_sync must be the WHOLE right-hand side of
// its own assignment (`val += __shfl_down_sync(mask, val, delta);` is
// fine -- the whole RHS of the += is the shuffle call). See task.md for
// the exact 5-step ladder.
__global__ void warp_final_reduce(float* out, const float* partial, int n) {
    int tid = threadIdx.x;
    float val = partial[tid];
    // TODO: 5-step shuffle-down ladder (delta 16, 8, 4, 2, 1), then
    // thread 0 writes out[0] = val.
}
