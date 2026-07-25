// Sum in[0..n) into out[0] using a balanced shared-memory tree reduction
// (every thread participates every round; round s combines sdata[tid] with
// sdata[tid+s] for tid < s, then halves s -- sequential addressing, no
// bank conflicts). One block, one thread per element (n == blockDim.x).
__global__ void reduce_sum(float* out, const float* in, int n) {
    int tid = threadIdx.x;
    // TODO: load in[tid] into __shared__ storage, tree-reduce it down to
    // a single total across log2(blockDim.x) rounds, then have thread 0
    // write the total to out[0].
    if (tid == 0) {
        out[0] = 0.0f;
    }
}
