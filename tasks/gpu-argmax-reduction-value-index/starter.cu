// Reduce in[0..n) to (max value, argmax index) and write out[0] = max
// value, out[1] = index of the max (lowest index on a tie), using a
// shared-memory tree reduction across the single block of n threads.
__global__ void argmax_reduce(const float* in, float* out, int n) {
    __shared__ float sval[32];
    __shared__ float sidx[32];
    int tid = threadIdx.x;
    // TODO: seed sval[tid]/sidx[tid] from in[tid]/tid, __syncthreads(),
    // then tree-reduce (stride 1, 2, 4, ... doubling) into sval[0]/sidx[0]
    // and have thread 0 write out[0]/out[1].
}
