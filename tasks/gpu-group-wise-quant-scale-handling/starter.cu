// Group-wise dequantization: elements are quantized in consecutive
// groups of G, each group carrying its own scale factor.
// out[i] = codes[i] * scale[i / G].
__global__ void dequant_groupwise(float* out, const float* codes, const float* scale, int G, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO: guard i < n, then out[i] = codes[i] * scale[i / G];
}
