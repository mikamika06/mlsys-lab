// Reference: each thread coarsens over 8 elements (base .. base+7), but
// processes them through a LOOP with ONE reused scalar temporary, so at
// most a handful of per-thread values are ever live at once, however many
// elements are coarsened.
__global__ void coarsened_square(float* out, const float* in, int n, float c) {
    int base = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    for (int k = 0; k < 8; k = k + 1) {
        float v = in[base + k];
        out[base + k] = v * v + c;
    }
}
