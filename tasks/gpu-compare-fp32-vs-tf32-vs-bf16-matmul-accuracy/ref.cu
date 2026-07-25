// Reference: matmul where every operand is round-to-nearest quantized to
// `mantissa_bits` explicit mantissa bits before multiplying (modeling
// fp32/tf32/bf16 truncation without needing real bitwise ops: rounding
// to n explicit mantissa bits is the same as rounding to the nearest
// multiple of 2^(exponent - n)). Accumulation itself stays at the
// simulator's native precision, exactly like a real tensor core
// accumulates in fp32 regardless of the input format.
__global__ void quantized_matmul(const float* A, const float* B, float* C,
                                  int N, float mantissa_bits) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * N) {
        int row = idx / N;
        int col = idx % N;
        float acc = 0.0;
        int k = 0;
        while (k < N) {
            float a = A[row * N + k];
            float asign = a < 0.0 ? -1.0 : 1.0;
            float aa = fabsf(a);
            float ae = floorf(logf(aa) / logf(2.0));
            float ascale = powf(2.0, ae - mantissa_bits);
            float aq = asign * floorf(aa / ascale + 0.5) * ascale;

            float b = B[k * N + col];
            float bsign = b < 0.0 ? -1.0 : 1.0;
            float bb = fabsf(b);
            float be = floorf(logf(bb) / logf(2.0));
            float bscale = powf(2.0, be - mantissa_bits);
            float bq = bsign * floorf(bb / bscale + 0.5) * bscale;

            acc = acc + aq * bq;
            k = k + 1;
        }
        C[idx] = acc;
    }
}
