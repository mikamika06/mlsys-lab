// Thread k packs codes[2k] (low nibble) and codes[2k+1] (high nibble),
// each in [0,16), into a single packed[k] slot as lo + hi*16, then
// immediately unpacks that same value back into roundtrip[2k]/roundtrip[2k+1].
__global__ void pack_unpack_int4(float* roundtrip, float* packed, const float* codes, int n) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO: guard 2*k+1 < n, then pack codes[2k]/codes[2k+1] into packed[k]
    // and unpack packed[k] back into roundtrip[2k]/roundtrip[2k+1].
}
