// Reference: online (streaming) softmax. First pass -- single-threaded,
// one value at a time -- maintains a running max `m` and running sum of
// exponentials `l`, RESCALING `l` every time the max changes (since every
// exponential already accumulated into `l` was computed relative to the
// OLD max): m_new = max(m, x_i); l = l*exp(m - m_new) + exp(x_i - m_new);
// m = m_new. After the stream, `m`/`l` are the exact same values a
// two-pass (max-then-sum) softmax would have produced. Second pass emits
// the normalized probabilities using the final m, l.
__global__ void online_softmax(float* out, const float* x, int n) {
    if (threadIdx.x == 0) {
        float m = x[0];
        float l = 1.0f;
        for (int i = 1; i < n; i++) {
            float xi = x[i];
            float m_new = fmaxf(m, xi);
            l = l * expf(m - m_new) + expf(xi - m_new);
            m = m_new;
        }
        for (int i = 0; i < n; i++) {
            out[i] = expf(x[i] - m) / l;
        }
    }
}
