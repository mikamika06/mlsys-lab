// Reference: both operands of every product are always rounded to
// fp16's 10-bit mantissa before multiplying (that's the "fp16 inputs"
// part, and it happens either way). When accumulate_fp16 is 0, the
// running sum `acc` stays at full precision -- true mixed-precision
// matmul, exactly what tensor cores do (fp16 x fp16 multiply, fp32
// accumulate). When accumulate_fp16 is 1, `acc` is ALSO rounded to
// fp16 mantissa after every single addition -- the naive "everything
// in fp16" baseline that mixed precision exists to avoid.
__global__ void mixed_precision_matmul(const float* A, const float* B, float* C,
                                        int N, int accumulate_fp16) {
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
            float ascale = powf(2.0, ae - 10.0);
            float aq = asign * floorf(aa / ascale + 0.5) * ascale;

            float b = B[k * N + col];
            float bsign = b < 0.0 ? -1.0 : 1.0;
            float bb = fabsf(b);
            float be = floorf(logf(bb) / logf(2.0));
            float bscale = powf(2.0, be - 10.0);
            float bq = bsign * floorf(bb / bscale + 0.5) * bscale;

            acc = acc + aq * bq;

            if (accumulate_fp16 > 0) {
                float accsign = acc < 0.0 ? -1.0 : 1.0;
                float aacc = fabsf(acc);
                if (aacc > 0.0) {
                    float acce = floorf(logf(aacc) / logf(2.0));
                    float accscale = powf(2.0, acce - 10.0);
                    acc = accsign * floorf(aacc / accscale + 0.5) * accscale;
                }
            }
            k = k + 1;
        }
        C[idx] = acc;
    }
}
