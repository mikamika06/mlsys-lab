__global__ void vector_copy_kernel(float* out, const float* in, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int base = tid * 4;
    if (base < n) out[base] = in[base];
    if (base + 1 < n) out[base + 1] = in[base + 1];
    if (base + 2 < n) out[base + 2] = in[base + 2];
    if (base + 3 < n) out[base + 3] = in[base + 3];
}
