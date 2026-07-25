// Reference: single-block shared-memory tree reduction that tracks BOTH
// the running max value and its index. Stride starts at 1 and DOUBLES
// (only threads where tid % (2*stride) == 0 stay active), so the active
// thread at position tid always merges with position tid+stride, whose
// provenance range [tid+stride, tid+2*stride) is entirely to the RIGHT
// of tid's own range [tid, tid+stride) -- so keeping tid's own value on
// a tie (sval[tid+stride] > sval[tid] is FALSE when equal) always keeps
// the LOWER original index.
__global__ void argmax_reduce(const float* in, float* out, int n) {
    __shared__ float sval[32];
    __shared__ float sidx[32];
    int tid = threadIdx.x;
    sval[tid] = in[tid];
    sidx[tid] = tid;
    __syncthreads();
    int stride = 1;
    while (stride < n) {
        if (tid % (2 * stride) == 0) {
            if (sval[tid + stride] > sval[tid]) {
                sval[tid] = sval[tid + stride];
                sidx[tid] = sidx[tid + stride];
            }
        }
        __syncthreads();
        stride = stride * 2;
    }
    if (tid == 0) {
        out[0] = sval[0];
        out[1] = sidx[0];
    }
}
