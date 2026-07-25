// Fill a 128-word __shared__ buffer from `seed` (every lane fills a
// strided subset of the 128 words), __syncthreads(), then each lane reads
// back exactly ONE shared word at offset idx[lane] and writes it to
// out[lane].
__global__ void bank_conflict_probe(float* out, const float* seed, const float* idx) {
    int lane = threadIdx.x;
    __shared__ float smem[128];
    // TODO: for j = lane; j < 128; j += 32: smem[j] = seed[j];
    // __syncthreads();
    // out[lane] = smem[(int)idx[lane]];
}
