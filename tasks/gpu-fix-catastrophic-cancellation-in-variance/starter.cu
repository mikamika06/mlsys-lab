// BUG: single-pass "textbook" variance formula E[x^2] - E[x]^2. Correct
// in exact arithmetic, but E[x^2] and E[x]^2 are both computed from the
// RAW values -- when the mean is large relative to the true spread, both
// terms are huge and nearly equal, and their difference loses almost all
// its significant digits (catastrophic cancellation). Fix it by
// centering: compute the mean first, then reduce (x - mean)^2 instead.
__global__ void row_variance(float* out, const float* x, int n) {
    __shared__ float ssum[32];
    __shared__ float ssq[32];
    int tid = threadIdx.x;
    float v = x[tid];
    ssum[tid] = v;
    ssq[tid] = v * v;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride = stride / 2) {
        if (tid < stride) {
            ssum[tid] = ssum[tid] + ssum[tid + stride];
            ssq[tid] = ssq[tid] + ssq[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        float mean = ssum[0] / n;
        float meansq = ssq[0] / n;
        out[0] = meansq - mean * mean;
    }
}
