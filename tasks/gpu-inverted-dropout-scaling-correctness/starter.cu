// Inverted dropout: out[i] = mask[i] * x[i] / (1 - p) for i in [0, n).
// mask[i] is 1.0 (keep) or 0.0 (drop) -- already computed for you. Keep the
// access coalesced: thread i must touch address i in x, mask, and out.
__global__ void dropout(float* out, const float* x, const float* mask, int n, float p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO: guard i < n, then out[i] = mask[i] * x[i] / (1.0f - p);
}
