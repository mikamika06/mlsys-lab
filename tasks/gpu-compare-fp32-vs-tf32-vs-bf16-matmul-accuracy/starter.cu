// TODO: for each output element C[row][col], sum over k of
// round(A[row][k], mantissa_bits) * round(B[k][col], mantissa_bits).
// To round v to `mantissa_bits` explicit mantissa bits: let
// e = floorf(logf(fabsf(v)) / logf(2.0)) (its binary exponent), let
// scale = powf(2.0, e - mantissa_bits), and round v to the nearest
// multiple of scale: sign(v) * floorf(fabsf(v)/scale + 0.5) * scale.
// See ref.cu's approach for the exact per-operand rounding shape.
__global__ void quantized_matmul(const float* A, const float* B, float* C,
                                  int N, float mantissa_bits) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * N) {
        C[idx] = 0.0;
    }
}
