// Sum 1024 floats (x[0..1023]) into out[0] using a shared-memory TREE
// (pairwise, stride-halving) reduction -- NOT a left-to-right sequential
// accumulation, which loses far more precision on wide-dynamic-range
// data. Load x[tid] into sdata[tid]; __syncthreads(); then for
// stride = 512, 256, ..., 1: if (tid < stride) sdata[tid] +=
// sdata[tid+stride]; followed by __syncthreads() every iteration; thread
// 0 writes out[0] = sdata[0]. See task.md.
__global__ void tree_sum(float* out, const float* x, int n) {
    __shared__ float sdata[1024];
    int tid = threadIdx.x;
    // TODO: load, barrier, stride-halving reduction loop (barrier every
    // step), thread 0 writes the result.
}
