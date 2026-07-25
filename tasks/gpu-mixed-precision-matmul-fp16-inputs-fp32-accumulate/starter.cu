// TODO: round both A[row][k] and B[k][col] to fp16's 10-bit mantissa
// before multiplying (always). Accumulate the products into `acc`. If
// accumulate_fp16 > 0, ALSO round `acc` itself to fp16 mantissa after
// every addition (guard against acc == 0 before taking its log).
// Otherwise leave `acc` at full precision. See ref.cu for the exact
// per-value rounding shape (sign * round(|v|/scale) * scale, with
// scale = 2^(floor(log2|v|) - 10)).
__global__ void mixed_precision_matmul(const float* A, const float* B, float* C,
                                        int N, int accumulate_fp16) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * N) {
        C[idx] = 0.0;
    }
}
