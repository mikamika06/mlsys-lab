// Compute out[row][col] = GELU((A @ B)[row][col] + bias[col]), fusing
// the bias-add and GELU activation directly into the matmul's epilogue:
// apply them to the accumulator register and write only the final
// activated value, without ever writing the raw matmul result to
// global memory. See task.md for the tanh-approximation GELU formula
// (this CUDA-C subset has no tanh builtin -- use its exp identity).
__global__ void matmul_bias_gelu(float* out, const float* A, const float* B, const float* bias, int M, int K, int N) {
    // TODO
}
