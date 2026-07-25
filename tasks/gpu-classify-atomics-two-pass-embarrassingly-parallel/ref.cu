// Reference: classify which inter-thread synchronization strategy an
// aggregation pattern needs, from two structural flags. Checking
// shared_target_unknown FIRST matters: a data-dependent collision needs
// atomics no matter what, and a global atomicAdd on device memory
// already reaches every block, so it subsumes the two-pass case -- it is
// NOT "atomics, then also two-pass".
__global__ void classify_sync_strategy(float* out, const float* cross_block,
                                        const float* shared_target_unknown, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (shared_target_unknown[i] > 0.5f) {
            out[i] = 1.0f;  // ATOMICS: collision target is data-dependent
        } else if (cross_block[i] > 0.5f) {
            out[i] = 2.0f;  // TWO-PASS: fixed targets, but needs a cross-block combine
        } else {
            out[i] = 0.0f;  // EMBARRASSINGLY PARALLEL: no collision, no cross-block need
        }
    }
}
