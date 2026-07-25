// Reference: accumulate the same stream of small increments two ways.
// out_fp32: plain running sum (stand-in for full-precision accumulation --
//           nothing is thrown away between additions).
// out_fp16: models fp16's limited mantissa AT THIS SPECIFIC MAGNITUDE --
//           half-precision floats have exactly a 1.0 ULP for any value in
//           [1024, 2048) (10 mantissa bits => 1024 distinct values per
//           octave), so every intermediate sum is rounded to the nearest
//           integer (floorf(x + 0.5f)) before the next addition, exactly
//           as an fp16 accumulator register would truncate it.
__global__ void accumulate_precision_demo(float* out_fp32, float* out_fp16,
                                           const float* base, const float* inc, int n) {
    if (threadIdx.x == 0) {
        float acc32 = base[0];
        float acc16 = base[0];
        for (int i = 0; i < n; i++) {
            acc32 = acc32 + inc[i];
            acc16 = floorf(acc16 + inc[i] + 0.5f);
        }
        out_fp32[0] = acc32;
        out_fp16[0] = acc16;
    }
}
