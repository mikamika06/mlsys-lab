// Fused pointwise kernel: for every i in [0, n), read x[i] ONCE and write
// BOTH activations from it.
//   SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
//   GELU(x) ~= 0.5*x*(1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
// There is no tanh() builtin in this CUDA-C subset -- build it from expf:
// tanh(z) = (exp(2z) - 1) / (exp(2z) + 1). sqrt(2/pi) ~= 0.7978845608.
__global__ void fused_gelu_silu(float* gelu_out, float* silu_out, const float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // TODO: load x[i] once, then write silu_out[i] and gelu_out[i]
        // using the formulas above.
    }
}
