// Reference: shared-memory tree reduction over 8 elements, one thread
// per element. A __syncthreads() after the initial load AND after every
// halving step ensures no thread ever reads a shared-memory slot before
// the thread responsible for writing it this step has actually done so.
__global__ void sum_reduce(float* out, const float* in, int n) {
    int tid = threadIdx.x;
    __shared__ float sdata[8];
    sdata[tid] = in[tid];
    __syncthreads();

    int stride = n / 2;
    while (stride > 0) {
        if (tid < stride) {
            sdata[tid] = sdata[tid] + sdata[tid + stride];
        }
        __syncthreads();
        stride = stride / 2;
    }

    if (tid == 0) {
        out[0] = sdata[0];
    }
}
