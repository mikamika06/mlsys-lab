// Reference: for each lane, derive the SOURCE lane a butterfly
// (XOR-shuffle) step with a given power-of-two `mask` reads from --
// `lane XOR mask`. This CUDA-C subset has no bitwise operators, so XOR
// with a power-of-two mask is computed arithmetically instead: bit
// `mask`'s value in `lane` is `(lane / mask) % 2` (integer division by a
// power of two isolates that bit, mod 2 reads it); if that bit is 0,
// XOR-ing it in ADDS mask to lane; if it's 1, XOR-ing it in SUBTRACTS
// mask from lane.
__global__ void shfl_xor_source_lane(float* out, int mask, int n) {
    int lane = threadIdx.x;
    if (lane < n) {
        int bit = (lane / mask) % 2;
        out[lane] = (bit == 0) ? lane + mask : lane - mask;
    }
}
