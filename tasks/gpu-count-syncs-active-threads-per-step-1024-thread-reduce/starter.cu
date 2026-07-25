// Sum the 1024 elements of `in` into out[0] using a single block of 1024
// threads and a tree reduction in shared memory: load in[tid] into
// sdata[tid]; then for stride = blockDim.x/2, /2, /2, ... down to 1,
// thread tid does sdata[tid] += sdata[tid+stride] only when tid < stride,
// with a __syncthreads() after every step (including the initial load);
// thread 0 writes out[0] = sdata[0] at the end.
__global__ void block_reduce_sum(float* out, const float* in, int n) {
    __shared__ float sdata[1024];
    int tid = threadIdx.x;
    // TODO: load, barrier, tree-reduction loop with a barrier each step,
    // then out[0] = sdata[0] from thread 0.
}
