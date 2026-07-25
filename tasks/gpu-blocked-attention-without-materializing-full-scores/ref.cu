// Reference: single-pass (online-softmax) attention. Thread i owns query
// row i and streams over ALL S keys/values one at a time, never storing
// more than a running max (m), running sum (l), and the running
// (unnormalized) output accumulator -- which lives directly in out[i*d
// .. i*d+d), read-modify-written every step -- so the full S x S score
// matrix is never materialized anywhere.
__global__ void flash_attn(float* out, const float* Q, const float* K, const float* V, int S, int d, int bk) {
    int i = threadIdx.x;
    if (i < S) {
        float scale = 1.0f / sqrtf(d);
        float m = -1e30f;
        float l = 0.0f;
        int dd = 0;
        for (dd = 0; dd < d; dd++) {
            out[i * d + dd] = 0.0f;
        }
        int kb = 0;
        for (kb = 0; kb < S; kb += bk) {
            int kb_end = kb + bk;
            if (kb_end > S) kb_end = S;
            int j = 0;
            for (j = kb; j < kb_end; j++) {
                float s = 0.0f;
                for (dd = 0; dd < d; dd++) {
                    s = s + Q[i * d + dd] * K[j * d + dd];
                }
                s = s * scale;
                float new_m = (s > m) ? s : m;
                float correction = expf(m - new_m);
                l = l * correction;
                for (dd = 0; dd < d; dd++) {
                    out[i * d + dd] = out[i * d + dd] * correction;
                }
                float p = expf(s - new_m);
                l = l + p;
                for (dd = 0; dd < d; dd++) {
                    out[i * d + dd] = out[i * d + dd] + p * V[j * d + dd];
                }
                m = new_m;
            }
        }
        for (dd = 0; dd < d; dd++) {
            out[i * d + dd] = out[i * d + dd] / l;
        }
    }
}
