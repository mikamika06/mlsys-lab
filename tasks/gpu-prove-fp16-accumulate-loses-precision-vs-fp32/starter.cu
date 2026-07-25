// Accumulate the same stream of increments two ways: out_fp32 as a plain
// running sum, out_fp16 rounding EVERY intermediate sum to the nearest
// integer (floorf(x + 0.5f)) before the next addition -- modeling fp16's
// exactly-1.0 ULP for values in [1024, 2048).
__global__ void accumulate_precision_demo(float* out_fp32, float* out_fp16,
                                           const float* base, const float* inc, int n) {
    // TODO: guard threadIdx.x == 0. acc32 = base[0], acc16 = base[0];
    // for i in [0,n): acc32 += inc[i]; acc16 = floorf(acc16 + inc[i] + 0.5f);
    // out_fp32[0] = acc32; out_fp16[0] = acc16;
}
