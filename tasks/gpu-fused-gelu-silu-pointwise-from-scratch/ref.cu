// Reference: one fused elementwise pass computing BOTH activations from
// the same loaded x[i] -- one global read feeds two independent outputs
// instead of two separate kernel launches each re-reading x.
//   SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
//   GELU(x) ~= 0.5*x*(1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
// tanh has no builtin in this CUDA-C subset, so it's built from expf:
// tanh(z) = (exp(2z) - 1) / (exp(2z) + 1).
__global__ void fused_gelu_silu(float* gelu_out, float* silu_out, const float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];

        float sig = 1.0f / (1.0f + expf(-v));
        silu_out[i] = v * sig;

        float inner = 0.7978845608f * (v + 0.044715f * v * v * v);
        float e2 = expf(2.0f * inner);
        float t = (e2 - 1.0f) / (e2 + 1.0f);
        gelu_out[i] = 0.5f * v * (1.0f + t);
    }
}
