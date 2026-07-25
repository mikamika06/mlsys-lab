// BUG: this reduction is missing a __syncthreads() between halving
// steps, so a thread can race ahead into the next step and read a
// shared-memory slot before the thread responsible for updating it this
// step has actually written it. Find the missing barrier and add it
// back. See task.md.
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
        stride = stride / 2;
    }

    if (tid == 0) {
        out[0] = sdata[0];
    }
}
