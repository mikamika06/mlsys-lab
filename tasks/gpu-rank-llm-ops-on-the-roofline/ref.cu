// Reference (single thread): compute arithmetic intensity (FLOPs/byte)
// for 4 canonical LLM ops, and classify each against the hardware's
// ridge point (peak_flops / peak_bandwidth):
//   GEMM (M,K,N):        flops = 2*M*K*N,        bytes = 4*(M*K + K*N + M*N)
//   attention (S,D):     flops = 4*S*S*D,         bytes = 4*(3*S*D + S*S + S*D)
//                         (Q,K,V loaded once each, one S*S score matrix,
//                         one S*D output -- QK^T and softmax*V are each 2*S*S*D flops)
//   LayerNorm (N elems): flops = 5*N (mean/var/normalize),  bytes = 4*2*N (read+write)
//   elementwise (N elems): flops = N,              bytes = 4*2*N (read+write)
__global__ void roofline_rank(float* out,
                                float gemm_m, float gemm_k, float gemm_n,
                                float attn_s, float attn_d,
                                float ln_n, float ew_n,
                                float peak_flops, float peak_bw) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        float ridge = peak_flops / peak_bw;

        float gemm_flops = 2.0f * gemm_m * gemm_k * gemm_n;
        float gemm_bytes = 4.0f * (gemm_m * gemm_k + gemm_k * gemm_n + gemm_m * gemm_n);
        float gemm_ai = gemm_flops / gemm_bytes;

        float attn_flops = 4.0f * attn_s * attn_s * attn_d;
        float attn_bytes = 4.0f * (3.0f * attn_s * attn_d + attn_s * attn_s + attn_s * attn_d);
        float attn_ai = attn_flops / attn_bytes;

        float ln_flops = 5.0f * ln_n;
        float ln_bytes = 4.0f * 2.0f * ln_n;
        float ln_ai = ln_flops / ln_bytes;

        float ew_flops = ew_n;
        float ew_bytes = 4.0f * 2.0f * ew_n;
        float ew_ai = ew_flops / ew_bytes;

        out[0] = gemm_ai;
        out[1] = attn_ai;
        out[2] = ln_ai;
        out[3] = ew_ai;

        out[4] = (gemm_ai >= ridge) ? 1.0f : 0.0f;
        out[5] = (attn_ai >= ridge) ? 1.0f : 0.0f;
        out[6] = (ln_ai >= ridge) ? 1.0f : 0.0f;
        out[7] = (ew_ai >= ridge) ? 1.0f : 0.0f;
    }
}
