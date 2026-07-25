// BROKEN: this online-softmax attention kernel rescales the running
// denominator `l` by alpha when merging in a new K-block, but NOT the
// running accumulator (acc0..acc3) -- so `l` and the accumulator drift
// out of consistent normalization the moment any block after the first
// raises the running max (alpha != 1). Find and fix the missing rescale.
__global__ void flash_attn_row(float* O, const float* Q, const float* K, const float* V,
                                int num_queries, int D, int num_keys, int block_k) {
    int i = threadIdx.x;
    if (i < num_queries) {
        float q0 = Q[i * D + 0];
        float q1 = Q[i * D + 1];
        float q2 = Q[i * D + 2];
        float q3 = Q[i * D + 3];
        float m = -1e30f;
        float l = 0.0f;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        float acc2 = 0.0f;
        float acc3 = 0.0f;
        int num_blocks = num_keys / block_k;
        for (int b = 0; b < num_blocks; b++) {
            float block_max = -1e30f;
            for (int jj = 0; jj < block_k; jj++) {
                int j = b * block_k + jj;
                float s = q0 * K[j * D + 0] + q1 * K[j * D + 1] + q2 * K[j * D + 2] + q3 * K[j * D + 3];
                block_max = fmaxf(block_max, s);
            }
            float new_m = fmaxf(m, block_max);
            float alpha = expf(m - new_m);

            float block_l = 0.0f;
            float block_acc0 = 0.0f;
            float block_acc1 = 0.0f;
            float block_acc2 = 0.0f;
            float block_acc3 = 0.0f;
            for (int jj = 0; jj < block_k; jj++) {
                int j = b * block_k + jj;
                float s = q0 * K[j * D + 0] + q1 * K[j * D + 1] + q2 * K[j * D + 2] + q3 * K[j * D + 3];
                float p = expf(s - new_m);
                block_l += p;
                block_acc0 += p * V[j * D + 0];
                block_acc1 += p * V[j * D + 1];
                block_acc2 += p * V[j * D + 2];
                block_acc3 += p * V[j * D + 3];
            }

            l = l * alpha + block_l;
            // BUG: the accumulator must ALSO be rescaled by alpha here,
            // exactly like `l` just was, before adding this block's
            // (already new_m-relative) contribution.
            acc0 = acc0 + block_acc0;
            acc1 = acc1 + block_acc1;
            acc2 = acc2 + block_acc2;
            acc3 = acc3 + block_acc3;
            m = new_m;
        }
        O[i * D + 0] = acc0 / l;
        O[i * D + 1] = acc1 / l;
        O[i * D + 2] = acc2 / l;
        O[i * D + 3] = acc3 / l;
    }
}
