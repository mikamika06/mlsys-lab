// For each i, classify the aggregation pattern described by
// cross_block[i] (1.0 if it needs a combine across MULTIPLE blocks, 0.0
// if it doesn't) and shared_target_unknown[i] (1.0 if the write target
// is data-dependent -- different threads can collide on the same output
// slot in a way not known ahead of time, 0.0 otherwise) into
// out[i]: 0.0 = embarrassingly parallel, 1.0 = atomics, 2.0 = two-pass.
// See task.md for the exact priority rule.
__global__ void classify_sync_strategy(float* out, const float* cross_block,
                                        const float* shared_target_unknown, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO: classify out[i] from cross_block[i] and shared_target_unknown[i]
}
