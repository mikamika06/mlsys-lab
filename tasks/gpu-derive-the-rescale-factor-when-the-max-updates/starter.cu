// Single-thread online-softmax scan over x[0..n). Track a running max
// `m`. For i == 0, factors[0] = 1.0 (no previous max yet). For i >= 1,
// derive the rescale factor that the running sum accumulated so far
// would need, given the max is about to move from `m` to
// max(m, x[i]).
__global__ void online_softmax_factors(float* factors, const float* x, int n) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        float m = x[0];
        // TODO: factors[0] = 1.0f; then for i in [1, n): compute
        // new_m = max(m, x[i]); factors[i] = expf(m - new_m); m = new_m;
    }
}
