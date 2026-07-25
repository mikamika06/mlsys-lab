// TODO: for each output C[i][j], sum over k of A[i][k] times the
// SYMMETRICALLY QUANTIZED-AND-DEQUANTIZED value of W[k][j]: round
// W[k][j]/scale to the nearest integer, clamp its magnitude to 127,
// then multiply back by scale. Fuse this quantize-dequantize step right
// into the matmul's accumulation loop. See ref.cu for the exact
// rounding shape.
__global__ void int8_dequant_matmul(const float* A, const float* W, float* C,
                                     int M, int N, int K, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N) {
        C[idx] = 0.0;
    }
}
