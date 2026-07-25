// Reference: fused int4-unpack + group-wise-dequant + matvec.
// packed_w[m, p] packs two int4 codes (each in [0,16)) for row m,
// columns 2p (low nibble) and 2p+1 (high nibble): b = lo + hi*16.
// Each column k's code is scaled by its row-and-group's own scale
// factor, scale[m, k/G], then dotted against x[k].
__global__ void dequant_matvec(float* y, const float* packed_w, const float* scale,
                                const float* x, int M, int K, int G) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m < M) {
        float acc = 0.0f;
        int half = K / 2;
        for (int p = 0; p < half; p++) {
            float b = packed_w[m * half + p];
            float hi = floorf(b / 16.0f);
            float lo = b - hi * 16.0f;
            int k0 = 2 * p;
            int k1 = 2 * p + 1;
            int g0 = k0 / G;
            int g1 = k1 / G;
            acc = acc + lo * scale[m * (K / G) + g0] * x[k0];
            acc = acc + hi * scale[m * (K / G) + g1] * x[k1];
        }
        y[m] = acc;
    }
}
