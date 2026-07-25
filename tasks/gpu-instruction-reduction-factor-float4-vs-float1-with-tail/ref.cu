// Reference (single thread): derive the instruction counts for reading n
// floats as scalar float1 loads versus vectorized float4 loads with a
// scalar tail for whatever doesn't fill a full float4.
__global__ void float4_instr_counts(float* out, int n) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int loads1 = n;
        int loads4 = n / 4 + n % 4;
        float loads1f = loads1 * 1.0f;
        float ratio = loads1f / loads4;
        out[0] = loads1;
        out[1] = loads4;
        out[2] = ratio;
    }
}
