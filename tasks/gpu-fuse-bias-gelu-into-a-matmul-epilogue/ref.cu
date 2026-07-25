// Reference: matmul with a FUSED bias + GELU epilogue. One thread per
// output element (row, col). Each thread reduces its own dot product
// entirely in a register, adds this column's bias, applies the
// tanh-approximation GELU (tanh implemented via its exact exponential
// identity, since this CUDA-C subset has no tanh builtin), and writes
// the FINAL activated value directly -- the raw matmul result
// (`row @ col + bias`) never touches global memory at all, let alone
// gets written out and read back before the activation runs.
__global__ void matmul_bias_gelu(float* out, const float* A, const float* B, const float* bias, int M, int K, int N) {
    int tid = threadIdx.x;
    int row = tid / N;
    int col = tid % N;

    float acc = 0.0f;
    for (int k = 0; k < K; k++) {
        acc = acc + A[row * K + k] * B[k * N + col];
    }
    float z = acc + bias[col];

    float c0 = 0.7978845608;
    float c1 = 0.044715;
    float u = c0 * (z + c1 * z * z * z);
    float e2u = expf(2.0 * u);
    float t = (e2u - 1.0) / (e2u + 1.0);  // tanh(u), via its exact exp identity
    float g = 0.5 * z * (1.0 + t);

    out[row * N + col] = g;
}
