// Reference: single-pass FlashAttention forward via the ONLINE
// (streaming) softmax recurrence -- one thread per query row, one key
// at a time, never materializing a full row of scores. head_dim is
// fixed at 4 (named scalars, not an array, since this language has no
// local arrays).
//
// Running state per query: m (max score seen so far), l (sum of
// exp(score - m) seen so far, rescaled), acc (weighted sum of V seen
// so far, rescaled). Each new key updates them via the standard
// online-softmax recurrence: whenever the running max increases from
// m to new_m, everything accumulated so far is corrected by a factor
// of exp(m - new_m) before the new term is added in.
__global__ void flash_attention_fwd(const float* Q, const float* K, const float* V,
                                     float* O, int N, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float q0 = Q[i * 4 + 0];
        float q1 = Q[i * 4 + 1];
        float q2 = Q[i * 4 + 2];
        float q3 = Q[i * 4 + 3];

        float m = -1.0e30;
        float l = 0.0;
        float acc0 = 0.0;
        float acc1 = 0.0;
        float acc2 = 0.0;
        float acc3 = 0.0;

        int j = 0;
        while (j < N) {
            float k0 = K[j * 4 + 0];
            float k1 = K[j * 4 + 1];
            float k2 = K[j * 4 + 2];
            float k3 = K[j * 4 + 3];
            float score = (q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3) * scale;

            float new_m = score > m ? score : m;
            float correction = expf(m - new_m);
            float p = expf(score - new_m);

            l = l * correction + p;

            float v0 = V[j * 4 + 0];
            float v1 = V[j * 4 + 1];
            float v2 = V[j * 4 + 2];
            float v3 = V[j * 4 + 3];

            acc0 = acc0 * correction + p * v0;
            acc1 = acc1 * correction + p * v1;
            acc2 = acc2 * correction + p * v2;
            acc3 = acc3 * correction + p * v3;

            m = new_m;
            j = j + 1;
        }

        O[i * 4 + 0] = acc0 / l;
        O[i * 4 + 1] = acc1 / l;
        O[i * 4 + 2] = acc2 / l;
        O[i * 4 + 3] = acc3 / l;
    }
}
