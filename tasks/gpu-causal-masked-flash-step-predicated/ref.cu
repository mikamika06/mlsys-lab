// Reference: one query per lane (s = 32 = one warp, d = 4), streaming
// online-softmax causal attention. The causal mask (j <= i) is applied by
// PREDICATION -- a 0/1 weight folded into the score -- never a branch, so
// every lane issues the exact same sequence of loads/stores and the warp
// never diverges. d = 4 is fixed, so the per-dimension accumulators are
// unrolled into named scalars (this CUDA-C subset has no local arrays).
__global__ void flash_step(const float* q, const float* k, const float* v, float* o,
                            int s, int d, float scale) {
    int i = threadIdx.x;

    float q0 = q[i * d + 0];
    float q1 = q[i * d + 1];
    float q2 = q[i * d + 2];
    float q3 = q[i * d + 3];

    float m_i = -1e30;
    float l_i = 0.0;
    float acc0 = 0.0;
    float acc1 = 0.0;
    float acc2 = 0.0;
    float acc3 = 0.0;

    for (int j = 0; j < s; j++) {
        float dot = q0 * k[j * d + 0] + q1 * k[j * d + 1] + q2 * k[j * d + 2] + q3 * k[j * d + 3];
        dot = dot * scale;

        float keep = (j <= i) ? 1.0 : 0.0;
        float score = dot * keep + (-1e30) * (1.0 - keep);

        float m_new = (m_i > score) ? m_i : score;
        float corr = expf(m_i - m_new);
        float w = expf(score - m_new);
        l_i = l_i * corr + w;

        acc0 = acc0 * corr + w * v[j * d + 0];
        acc1 = acc1 * corr + w * v[j * d + 1];
        acc2 = acc2 * corr + w * v[j * d + 2];
        acc3 = acc3 * corr + w * v[j * d + 3];

        m_i = m_new;
    }

    o[i * d + 0] = acc0 / l_i;
    o[i * d + 1] = acc1 / l_i;
    o[i * d + 2] = acc2 / l_i;
    o[i * d + 3] = acc3 / l_i;
}
