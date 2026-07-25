// Reference: numerically SAFE row-wise softmax. Subtracting the row's
// own max logit before exponentiating guarantees every exponent is
// <= 0, so expf() never overflows -- no matter how large the logits
// themselves are -- and the result is mathematically identical to the
// naive softmax(x) = exp(x) / sum(exp(x)), since that constant shift
// cancels exactly in the ratio.
__global__ void safe_softmax_row(float* out, const float* logits, int n_rows, int D) {
    int i = threadIdx.x;
    if (i < n_rows) {
        float m = -1e30f;
        for (int d = 0; d < D; d++) {
            float v = logits[i * D + d];
            m = fmaxf(m, v);
        }
        float sum = 0.0f;
        for (int d = 0; d < D; d++) {
            float p = expf(logits[i * D + d] - m);
            sum += p;
        }
        for (int d = 0; d < D; d++) {
            float p = expf(logits[i * D + d] - m);
            out[i * D + d] = p / sum;
        }
    }
}
