// Roofline classification: out[i] = 1.0 if ai[i] >= ridge (compute-bound),
// else 0.0 (memory-bound). One element per thread; keep it coalesced --
// thread i must touch address i in both `ai` and `out`.
__global__ void classify(float* out, const float* ai, int n, float ridge) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO: guard i < n, then out[i] = (ai[i] >= ridge) ? 1.0f : 0.0f;
}
