// Softmax over each 32-lane warp's row, entirely via warp shuffle -- no
// shared memory. Two 5-step __shfl_xor_sync butterfly ladders (delta =
// 16,8,4,2,1): first max-reduce `val` (fmaxf at each step) to get the
// row max `m` broadcast to every lane; then sum-reduce `expf(val - m)`
// to get the row sum `s`, also broadcast to every lane. Each
// __shfl_xor_sync must be the whole right-hand side of its own
// assignment. out[tid] = expf(val - m) / s.
__global__ void warp_softmax(float* out, const float* x, int n) {
    int tid = threadIdx.x;
    float val = x[tid];
    // TODO: 5-step shfl_xor max-reduction ladder for m, then 5-step
    // shfl_xor sum-reduction ladder (over expf(val - m)) for s, then
    // out[tid] = expf(val - m) / s.
}
