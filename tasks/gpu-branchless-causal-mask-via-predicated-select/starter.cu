// Branchy causal mask, produces the RIGHT VALUES but issues a different
// number of memory accesses per thread depending on the mask: a thread
// with j <= i loads score[idx] then stores; a thread with j > i only
// stores. Rewrite this branchlessly: EVERY thread must always load
// score[idx] and always store out[idx], folding the mask into the value
// with a 0/1 predicate instead of an if/else around the memory ops.
__global__ void causal_mask(float* out, const float* score, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = idx / n;
    int j = idx % n;
    if (j <= i) {
        out[idx] = score[idx];
    } else {
        out[idx] = -1.0e30f;
    }
}
