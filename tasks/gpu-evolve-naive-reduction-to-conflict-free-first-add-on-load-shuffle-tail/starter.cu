// VALID but unoptimized: Harris's "kernel 1" -- one thread per element
// (no first-add-on-load), interleaved addressing with a modulo condition
// (bank-conflict-prone, and divergent within a warp for every stride
// after the first), and no shuffle tail: every single halving, all the
// way down to tid == 0, goes through shared memory and __syncthreads().
// Gets the right answer, at a much higher shared-memory and cycle cost.
__global__ void reduce_sum(float* out, const float* x, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    sdata[tid] = x[tid];
    __syncthreads();

    int stride = 1;
    while (stride < 256) {
        if (tid % (2 * stride) == 0) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
        stride = stride * 2;
    }

    if (tid == 0) {
        out[0] = sdata[0];
    }
}
