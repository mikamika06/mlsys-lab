// Reference: privatized shared-memory histogram with one atomic flush per
// bin per block. Each block first bins its own slice of `input` into a
// PRIVATE per-block histogram in shared memory -- contention there is only
// ever between the (up to) 32 threads of the SAME block, and only when two
// of them land in the same bin, so that accumulate must be atomic. Then
// exactly one thread per bin does ONE atomicAdd to flush that block's
// partial counts into the global histogram -- contention there is between
// different BLOCKS' thread 0..7, all of whom may target the same global
// bin, so that flush must be atomic too.
__global__ void histogram_privatized(const float* input, float* out, int n) {
    __shared__ float hist[8];
    int tid = threadIdx.x;
    if (tid < 8) { hist[tid] = 0.0f; }
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int bin = (int)input[i];
        atomicAdd(&hist[bin], 1.0f);
    }
    __syncthreads();

    if (tid < 8) {
        atomicAdd(&out[tid], hist[tid]);
    }
}
