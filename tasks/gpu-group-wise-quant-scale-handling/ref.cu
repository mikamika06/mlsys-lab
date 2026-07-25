// Reference: group-wise dequantization. Elements are quantized in
// consecutive groups of G, each group carrying its OWN scale factor:
// out[i] = codes[i] * scale[i / G].
__global__ void dequant_groupwise(float* out, const float* codes, const float* scale, int G, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int g = i / G;
        out[i] = codes[i] * scale[g];
    }
}
