// Reference: single-block tree reduction with SEQUENTIAL addressing.
// stride starts at blockDim.x/2 and halves each step; only thread
// tid < stride is active. At every stride >= 32, all 8 warps are either
// fully active or fully inactive (no warp is ever half-in/half-out), and
// thread tid's two shared-memory words (tid and tid+stride) always land
// in DIFFERENT banks from every other active thread's pair -- zero bank
// conflicts throughout.
__global__ void block_reduce_sum(float* out, const float* in, int n) {
    __shared__ float sdata[256];
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
