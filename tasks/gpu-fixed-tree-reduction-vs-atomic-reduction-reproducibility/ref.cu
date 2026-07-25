// Reference: two reduction strategies over the same 32 values.
//
// tree_reduce_sum: a fixed-structure sequential-addressing tree
// reduction. Which PAIR of shared-memory slots gets combined at every
// step is determined entirely by thread index and stride -- never by
// which thread happened to run first -- so the combine order (and hence
// the exact rounding at every step) is always identical, run after run.
//
// ordered_reduce_sum: models what a chain of atomicAdd()s would produce
// if the adds were granted to threads in the sequence given by `order`
// (a permutation of [0, n)) -- single-threaded, purely sequential, one
// float32-style add at a time in that exact sequence.
__global__ void tree_reduce_sum(float* out, const float* x, int n) {
    __shared__ float sdata[32];
    int tid = threadIdx.x;
    sdata[tid] = x[tid];
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride = stride / 2) {
        if (tid < stride) {
            sdata[tid] = sdata[tid] + sdata[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[0] = sdata[0];
    }
}

__global__ void ordered_reduce_sum(float* out, const float* x, const float* order, int n) {
    if (threadIdx.x == 0) {
        float acc = 0.0f;
        for (int k = 0; k < n; k++) {
            int idx = order[k];
            acc = acc + x[idx];
        }
        out[0] = acc;
    }
}
