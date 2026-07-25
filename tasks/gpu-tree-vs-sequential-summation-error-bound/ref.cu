// Reference: sum 1024 floats with a shared-memory TREE (pairwise,
// stride-halving) reduction. Every rounding step in a tree reduction
// combines two partial sums of roughly comparable size that are already
// close to their final magnitude, so the reduction's error grows only
// with its DEPTH -- O(log2 n) -- rather than with n itself. This matters
// most for wide-dynamic-range data: one huge value and many tiny ones.
// Naive left-to-right accumulation adds every tiny value directly into
// an already-large running total, one at a time -- each addition below
// the running total's own rounding granularity vanishes completely,
// losing that value's entire contribution, forever, before it ever gets
// a chance to combine with anything its own size.
__global__ void tree_sum(float* out, const float* x, int n) {
    __shared__ float sdata[1024];
    int tid = threadIdx.x;
    sdata[tid] = x[tid];
    __syncthreads();
    for (int stride = 512; stride > 0; stride /= 2) {
        if (tid < stride) {
            sdata[tid] = sdata[tid] + sdata[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[0] = sdata[0];
    }
}
