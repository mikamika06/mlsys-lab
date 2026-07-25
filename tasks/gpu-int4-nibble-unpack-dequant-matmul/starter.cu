// Fused int4-unpack + group-wise-dequant + matvec. packed_w[m, p] packs
// two int4 codes for row m, columns 2p (low nibble) and 2p+1 (high
// nibble): b = lo + hi*16. Column k's code is scaled by scale[m, k/G]
// (its row-and-group's own scale), then dotted against x[k], summed over
// all K columns into y[m].
__global__ void dequant_matvec(float* y, const float* packed_w, const float* scale,
                                const float* x, int M, int K, int G) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO: guard m < M. For p = 0 .. K/2-1: unpack packed_w[m*(K/2)+p]
    // into lo/hi (floorf(b/16), b - hi*16); accumulate
    // lo*scale[m*(K/G)+(2p)/G]*x[2p] + hi*scale[m*(K/G)+(2p+1)/G]*x[2p+1].
    // Write the total to y[m].
}
