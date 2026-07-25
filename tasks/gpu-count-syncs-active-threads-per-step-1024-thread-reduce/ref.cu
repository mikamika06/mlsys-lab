// Reference: single-block, 1024-thread tree reduction. stride starts at
// blockDim.x/2 and halves each step (10 steps total, since 2^10 == 1024);
// only threads with tid < stride are active at a given step, and every
// step ends with __syncthreads() because the next step reads values the
// other active threads just wrote.
__global__ void block_reduce_sum(float* out, const float* in, int n) {
    __shared__ float sdata[1024];
    int tid = threadIdx.x;
    sdata[tid] = in[tid];
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[0] = sdata[0];
    }
}
