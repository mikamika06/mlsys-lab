// Reference: pack two int4 codes (each in [0,16)) into one byte-value
// slot, then immediately unpack that same slot back out, per thread.
// Thread k owns codes[2k] (low nibble) and codes[2k+1] (high nibble):
//   packed[k]      = lo + hi*16
//   roundtrip[2k]   = packed[k] - floor(packed[k]/16)*16   (recovered lo)
//   roundtrip[2k+1] = floor(packed[k]/16)                  (recovered hi)
__global__ void pack_unpack_int4(float* roundtrip, float* packed, const float* codes, int n) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (2 * k + 1 < n) {
        float lo = codes[2 * k];
        float hi = codes[2 * k + 1];
        float b = lo + hi * 16.0f;
        packed[k] = b;

        float hi2 = floorf(b / 16.0f);
        float lo2 = b - hi2 * 16.0f;
        roundtrip[2 * k] = lo2;
        roundtrip[2 * k + 1] = hi2;
    }
}
