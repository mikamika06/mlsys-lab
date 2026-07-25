// Reference: mixed-precision QK^T, modeling what a tensor core does --
// round EACH input to fp16 precision before multiplying, but accumulate
// the products in full precision (standing in for a real tensor core's
// fp32 accumulator). Q and K are restricted to [1,2) for this task,
// where fp16's 10 explicit mantissa bits give an EXACT ULP of 1/1024 --
// floorf(x*1024+0.5)/1024 is genuinely what fp16 rounding does to a
// value in that range, not an approximation.
// S[i][j] = (1/sqrt(D)) * sum_d fp16(Q[i][d]) * fp16(K[j][d]).
__global__ void qkt_mixed_precision(float* S, const float* Q, const float* K, int M, int N, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N) {
        int i = idx / N;
        int j = idx % N;
        float scale = 1.0f / sqrtf(D);
        float acc = 0.0f;
        for (int d = 0; d < D; d++) {
            float q = Q[i * D + d];
            float k = K[j * D + d];
            float qh = floorf(q * 1024.0f + 0.5f) / 1024.0f;
            float kh = floorf(k * 1024.0f + 0.5f) / 1024.0f;
            acc = acc + qh * kh;
        }
        S[idx] = scale * acc;
    }
}
