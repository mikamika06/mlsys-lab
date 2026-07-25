// Mixed-precision QK^T: S[i][j] = (1/sqrt(D)) * sum_d fp16(Q[i][d]) *
// fp16(K[j][d]), where fp16(x) = floorf(x*1024.0f + 0.5f) / 1024.0f
// (exact fp16 rounding for x in [1,2), which is the range Q and K are
// guaranteed to be in for this task). Accumulate the products in full
// precision, only round the INPUTS before each multiply.
__global__ void qkt_mixed_precision(float* S, const float* Q, const float* K, int M, int N, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO: guard idx < M*N. i = idx/N, j = idx%N. Accumulate
    // fp16(Q[i*D+d]) * fp16(K[j*D+d]) over d in [0,D), then
    // S[idx] = acc / sqrtf(D).
}
