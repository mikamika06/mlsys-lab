// Reference: each thread first cooperatively fills a 128-word shared
// buffer from `seed` (every lane fills a strided subset), then each lane
// reads back ONE shared word at a lane-specific offset given by `idx`
// (the fixed access pattern being probed for bank conflicts) and writes
// it to `out`.
__global__ void bank_conflict_probe(float* out, const float* seed, const float* idx) {
    int lane = threadIdx.x;
    __shared__ float smem[128];
    for (int j = lane; j < 128; j = j + 32) {
        smem[j] = seed[j];
    }
    __syncthreads();
    int i = idx[lane];
    out[lane] = smem[i];
}
