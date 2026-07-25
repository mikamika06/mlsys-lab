// BUG: interleaved addressing with a growing stride multiplier.
// index = 2*stride*tid picks out active threads WITHOUT a divergent
// modulo branch, but within a warp (32 consecutive tid), index mod 32
// only takes blockDim.x/(2*stride)... in practice: at stride=1, 32
// threads' indices (0,2,4,...,62) alias onto just 16 banks -- 2-way
// conflicts. At stride=2, indices (0,4,8,...,124) alias onto 8 banks --
// 4-way conflicts. It keeps doubling every step. Still computes the
// exactly correct sum -- fix the ADDRESSING, not the arithmetic.
__global__ void block_reduce_sum(float* out, const float* in, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    sdata[tid] = in[tid];
    __syncthreads();
    for (int stride = 1; stride < blockDim.x; stride = stride * 2) {
        int index = 2 * stride * tid;
        if (index < blockDim.x) {
            sdata[index] = sdata[index] + sdata[index + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[0] = sdata[0];
    }
}
