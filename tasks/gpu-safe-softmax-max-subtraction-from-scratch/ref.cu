// Reference: numerically-safe softmax, three single-threaded passes.
// Pass 1: find the row max m. Pass 2: sum exp(x_i - m). Pass 3: emit
// exp(x_i - m) / sum. Subtracting the max before exponentiating keeps
// every exponent <= 0, so exp() never overflows even when a raw logit
// is large enough that exp(logit) alone would.
__global__ void safe_softmax(float* out, const float* x, int n) {
    if (threadIdx.x == 0) {
        float m = x[0];
        for (int i = 1; i < n; i++) {
            m = fmaxf(m, x[i]);
        }
        float s = 0.0f;
        for (int i = 0; i < n; i++) {
            s = s + expf(x[i] - m);
        }
        for (int i = 0; i < n; i++) {
            out[i] = expf(x[i] - m) / s;
        }
    }
}
