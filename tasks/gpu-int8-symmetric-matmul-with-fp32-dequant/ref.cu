// Reference: dequant-fused int8 matmul. W is quantized symmetrically
// (zero-point 0) with a single per-tensor `scale`: each element rounds
// to the nearest integer multiple of `scale`, clamped to the int8 range
// [-127, 127] * scale. A stays full precision. Every element of W is
// quantized and immediately dequantized right where it's used, fused
// into the matmul's inner loop -- no separate quantize/dequantize pass.
__global__ void int8_dequant_matmul(const float* A, const float* W, float* C,
                                     int M, int N, int K, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N) {
        int i = idx / N;
        int j = idx % N;
        float acc = 0.0;
        int k = 0;
        while (k < K) {
            float w = W[k * N + j];
            float sign = w < 0.0 ? -1.0 : 1.0;
            float q = floorf(fabsf(w) / scale + 0.5);
            if (q > 127.0) {
                q = 127.0;
            }
            float dequant_w = sign * q * scale;
            acc = acc + A[i * K + k] * dequant_w;
            k = k + 1;
        }
        C[idx] = acc;
    }
}
