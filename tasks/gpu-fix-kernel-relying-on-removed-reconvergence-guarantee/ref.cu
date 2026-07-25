// Reference: the divergent branch's extra work is confined to the `if`,
// but the synchronization point is UNCONDITIONAL -- every lane, whichever
// branch it took, reaches the same __syncthreads() before any lane
// reaches the shuffle. That guarantees all 32 lanes arrive at the
// shuffle together, so `__shfl_up_sync` reads what its neighbor lane
// actually holds right now, not a stale/absent value from a neighbor
// that hasn't caught up yet.
__global__ void divergent_shuffle(float* out, const float* in, int n) {
    int tid = threadIdx.x;
    float val = in[tid];
    if (tid < 16) {
        val = val * 2.0;
    }
    __syncthreads();
    float shuffled = __shfl_up_sync(0xffffffff, val, 1);
    out[tid] = shuffled;
}
