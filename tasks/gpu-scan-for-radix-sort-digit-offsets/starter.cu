// TODO: (1) histogram `digits[i]` into hist[0..num_digits); (2)
// exclusive-scan hist into offsets (offsets[0]=0, offsets[d] =
// offsets[d-1] + hist[d-1]); (3) reset hist to 0 and reuse it as a
// per-digit cursor while walking the input IN ORDER, scattering
// out[offsets[dg] + hist[dg]] = keys[i] then incrementing hist[dg] --
// walking in original order is what keeps the scatter stable. See
// ref.cu.
__global__ void radix_scatter(const float* keys, const float* digits, float* out,
                               int n, int num_digits) {
    __shared__ float hist[16];
    __shared__ float offsets[16];
    // your code here
}
