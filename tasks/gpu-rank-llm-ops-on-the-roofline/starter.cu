// Single thread. Compute arithmetic intensity (out[0..3]) for GEMM,
// attention, LayerNorm, and elementwise (in that order), then classify
// each as compute-bound (1.0, out[4..7]) or memory-bound (0.0) against
// ridge = peak_flops / peak_bw.
__global__ void roofline_rank(float* out,
                                float gemm_m, float gemm_k, float gemm_n,
                                float attn_s, float attn_d,
                                float ln_n, float ew_n,
                                float peak_flops, float peak_bw) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // your code here
    }
}
