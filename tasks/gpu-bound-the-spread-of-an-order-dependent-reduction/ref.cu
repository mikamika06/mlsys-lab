// Reference: balanced shared-memory tree reduction. sdata[tid] starts as
// in[tid]; each round halves the number of live partial sums by adding the
// upper half onto the lower half (sequential addressing: no bank
// conflicts). After log2(blockDim.x) rounds, sdata[0] holds the total and
// thread 0 writes it out. One block, one thread per element (n ==
// blockDim.x).
__global__ void reduce_sum(float* out, const float* in, int n) {
    __shared__ float sdata[128];
    int tid = threadIdx.x;
    sdata[tid] = in[tid];
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s = s / 2) {
        if (tid < s) {
            sdata[tid] = sdata[tid] + sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[0] = sdata[0];
    }
}
