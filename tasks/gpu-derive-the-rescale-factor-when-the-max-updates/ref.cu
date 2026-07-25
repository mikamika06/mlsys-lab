// Reference: single-thread online-softmax scan. Maintains a running max
// `m` and, for every element after the first, derives the correction
// factor exp(old_m - new_m) that the running (unnormalized) sum would
// need to be rescaled by before adding this element's own exp(x[i] -
// new_m) contribution. factors[0] is the base case (no previous max, no
// correction: 1.0).
__global__ void online_softmax_factors(float* factors, const float* x, int n) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        float m = x[0];
        factors[0] = 1.0f;
        for (int i = 1; i < n; i++) {
            float new_m = m > x[i] ? m : x[i];
            factors[i] = expf(m - new_m);
            m = new_m;
        }
    }
}
