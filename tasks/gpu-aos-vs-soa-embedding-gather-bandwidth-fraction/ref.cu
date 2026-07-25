// Reference: gather each thread's embedding row and sum its D dimensions.
// `emb` is stored SoA (dimension-major): emb[d*V + v] holds dimension d of
// vocabulary row v, so all V rows of one dimension are contiguous. For a
// fixed loop step d, thread i's address is d*V + idx[i] -- with idx[i] = i
// across the warp, that's 32 CONSECUTIVE elements: one coalesced 128-byte
// transaction per step, instead of one per lane.
__global__ void gather_soa(float* out, const float* emb, const int* idx, int D, int V) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int v = idx[i];
    float sum = 0.0f;
    for (int d = 0; d < D; d++) {
        sum = sum + emb[d * V + v];
    }
    out[i] = sum;
}
