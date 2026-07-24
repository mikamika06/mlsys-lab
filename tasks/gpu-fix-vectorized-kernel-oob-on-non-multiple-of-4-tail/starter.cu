// BUG: each thread unconditionally copies 4 consecutive elements (a stand-in
// for a float4 vectorized load/store), with no bounds check against n. The
// last thread's group runs past the end of the array whenever n is not a
// multiple of 4.
__global__ void vector_copy_kernel(float* out, const float* in, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int base = tid * 4;
    out[base] = in[base];
    out[base + 1] = in[base + 1];
    out[base + 2] = in[base + 2];
    out[base + 3] = in[base + 3];
}
